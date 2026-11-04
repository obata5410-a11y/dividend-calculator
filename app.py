from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import io
import os
import re
import uuid
import time
from datetime import datetime, timezone
from collections import Counter

_HTTP = requests.Session()
_HTTP.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ja,en-US;q=0.9",
})

# 一時セッションストア（プロセス内メモリ、再起動でクリア）
_sessions: dict[str, dict] = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/result")
def result_page():
    return FileResponse("static/result.html")


def normalize_ticker(code: str) -> str:
    """銘柄コードをyfinance形式に正規化する。数字4桁なら日本株(.T付与)"""
    code = str(code).strip().upper()
    if re.fullmatch(r"\d{4}", code):
        return code + ".T"
    return code


def _fetch_jp_ticker_data(code: str) -> dict:
    """minkabu.jp から日本株の配当利回りと銘柄名を取得する。"""
    result = {"div_yield": None, "name": None}
    try:
        r = _HTTP.get(f"https://minkabu.jp/stock/{code}/dividend", timeout=10)
        if r.status_code != 200:
            return result
        soup = BeautifulSoup(r.text, "html.parser")

        # 配当利回り: meta description から抽出（最も安定）
        meta = soup.find("meta", attrs={"name": "description"})
        if meta:
            m = re.search(r"配当利回り[：:]\s*([\d.]+)%", meta.get("content", ""))
            if m:
                result["div_yield"] = float(m.group(1)) / 100

        # 銘柄名: title タグ "会社名 (code) : 配当情報 - みんかぶ" から抽出
        title = soup.find("title")
        if title:
            # "ＮＴＴ (9432) : 配当情報 [NTT] - みんかぶ" → "ＮＴＴ"
            m2 = re.match(r"^(.+?)\s*[（(]\d{4,5}[）)]", title.get_text(strip=True))
            if m2:
                result["name"] = m2.group(1).strip()

    except Exception:
        pass
    return result


def _fetch_us_ticker_data(ticker: str) -> dict:
    """Yahoo Finance v8 chart API に直接アクセスして米国株・ETFの配当利回りと銘柄名を取得する。"""
    result = {"div_yield": None, "name": None}
    try:
        r = _HTTP.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"range": "1y", "interval": "1mo", "events": "div"},
            headers={
                "Accept": "application/json",
                "Referer": f"https://finance.yahoo.com/quote/{ticker}",
            },
            timeout=15,
        )
        if r.status_code != 200:
            return result

        chart = r.json().get("chart", {}).get("result", [])
        if not chart:
            return result

        meta = chart[0].get("meta", {})
        price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        result["name"] = meta.get("longName") or meta.get("shortName")

        events = chart[0].get("events", {})
        divs = events.get("dividends", {})
        annual_div = sum(v["amount"] for v in divs.values())

        if annual_div > 0 and price and price > 0:
            result["div_yield"] = annual_div / price

    except Exception:
        pass
    return result


_fx_cache: dict = {"rate": None, "ts": 0.0}
FX_CACHE_TTL = 600  # 秒


def fetch_usd_jpy_rate() -> float | None:
    """Yahoo Finance v8 chart API から現在のドル円レートを取得する（TTLキャッシュ付き）。"""
    now = time.time()
    if _fx_cache["rate"] is not None and now - _fx_cache["ts"] < FX_CACHE_TTL:
        return _fx_cache["rate"]
    try:
        r = _HTTP.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/JPY=X",
            params={"range": "1d", "interval": "1m"},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if r.status_code == 200:
            chart = r.json().get("chart", {}).get("result", [])
            if chart:
                price = chart[0].get("meta", {}).get("regularMarketPrice")
                if price:
                    _fx_cache["rate"] = price
                    _fx_cache["ts"] = now
    except Exception:
        pass
    return _fx_cache["rate"]


def fetch_ticker_data(ticker: str) -> dict:
    """日本株（4桁.T）はminkabu優先、配当利回り未取得時はYahoo Finance v8にフォールバック。"""
    m = re.fullmatch(r"(\d{4})\.T", ticker)
    if m:
        result = _fetch_jp_ticker_data(m.group(1))
        if result.get("div_yield") is None:
            # 配当未発表などでminkabuが---を返す場合はYahoo Finance v8で補完
            yf = _fetch_us_ticker_data(ticker)
            if yf.get("div_yield") is not None:
                result["div_yield"] = yf["div_yield"]
            if result.get("name") is None and yf.get("name"):
                result["name"] = yf["name"]
        return result
    return _fetch_us_ticker_data(ticker)


def get_display_name(code: str, raw_name: str, fetched_name: str | None) -> str:
    """銘柄名がコード番号だけの場合にスクレイピング結果で補完する。"""
    if raw_name and raw_name != code and not re.fullmatch(r"[\dA-Z.]+", raw_name.strip()):
        return raw_name
    return fetched_name if fetched_name else raw_name


# ── 税率定数 ────────────────────────────────────────────────
# 日本株: 所得税15.315% + 住民税5% = 20.315%
TAX_JP = 0.20315
# 米国株: 米国源泉10% + 国内20.315%（外国税額控除なしの最大値）≒ 28.28%
#   計算: 1 - (1-0.10)*(1-0.20315) = 1 - 0.9*0.79685 ≈ 0.2828
TAX_US = round(1 - (1 - 0.10) * (1 - 0.20315), 6)
# 米国株をNISAで保有する場合も、米国での源泉徴収10%は残る
TAX_US_NISA = 0.10


def tax_rate_for(ticker: str) -> float:
    """ティッカーに応じた源泉徴収税率を返す。"""
    return TAX_JP if ticker.endswith(".T") else TAX_US


def nisa_tax_rate_for(ticker: str) -> float:
    """NISA口座の税率。国内銘柄は0%、米国銘柄は現地源泉10%。"""
    return 0.0 if ticker.endswith(".T") else TAX_US_NISA


# ── NISA（非課税）対象の銘柄 ────────────────────────────────
_NISA_FILE = os.path.join(os.path.dirname(__file__), "nisa_codes.txt")
_TICKER_ALIASES_FILE = os.path.join(os.path.dirname(__file__), "ticker_aliases.txt")
_YIELD_OVERRIDES_FILE = os.path.join(os.path.dirname(__file__), "yield_overrides.txt")


def _normalize_nisa_code(code: str) -> str:
    """NISA判定用にコードを正規化する（大文字化・末尾.T除去）。"""
    code = str(code).strip().upper()
    if code.endswith(".T"):
        code = code[:-2]
    return code


def load_nisa_lot_counts() -> Counter[str]:
    """nisa_codes.txt からコードごとのNISA保有行数を読み込む。

    同じコードを複数回記載すると、その件数分の保有行をNISA扱いにする。
    ファイルが無い・読めない場合は空Counter（＝全銘柄を課税扱い）。
    リクエストごとに読み込むため、編集後の再起動は不要。
    """
    codes: Counter[str] = Counter()
    try:
        with open(_NISA_FILE, encoding="utf-8") as f:
            for line in f:
                # 行頭コメント・空行・インラインコメント（コード横の # 以降）を除去
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                codes[_normalize_nisa_code(line)] += 1
    except OSError:
        pass
    return codes


def _normalize_config_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def _load_key_value_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() and value.strip():
                    values[_normalize_config_key(key)] = value.strip()
    except OSError:
        pass
    return values


def load_ticker_aliases() -> dict[str, str]:
    """コードが空の銘柄名をティッカーへ変換する設定を読み込む。"""
    return _load_key_value_file(_TICKER_ALIASES_FILE)


def load_yield_overrides() -> dict[str, float]:
    """銘柄名・コードごとの配当利回り上書き値（%）を小数へ変換する。"""
    result: dict[str, float] = {}
    for key, value in _load_key_value_file(_YIELD_OVERRIDES_FILE).items():
        try:
            result[key] = float(value.replace("%", "").strip()) / 100
        except ValueError:
            continue
    return result


def resolve_ticker(code: str, name: str, aliases: dict[str, str]) -> tuple[str, str]:
    """CSVのコード、または銘柄名エイリアスから表示コードとティッカーを返す。"""
    clean_code = str(code or "").strip()
    if clean_code.lower() == "nan" or clean_code == "-":
        clean_code = ""
    resolved_code = clean_code or aliases.get(_normalize_config_key(name), "")
    ticker = normalize_ticker(resolved_code) if resolved_code else ""
    return resolved_code, ticker


def get_dividend_yield(
    code: str,
    ticker: str,
    name: str,
    fetched_yield: float | None,
    overrides: dict[str, float],
) -> tuple[float | None, str]:
    """上書き値を優先し、なければ市場データの利回りを返す。"""
    for candidate in (code, ticker, name):
        key = _normalize_config_key(candidate)
        if key and key in overrides:
            return overrides[key], "override"
    return fetched_yield, "market" if fetched_yield is not None else "unavailable"


def calc_after_tax(gross: float | None, rate: float) -> float | None:
    return round(gross * (1 - rate), 0) if gross is not None else None


@app.post("/api/calculate")
async def calculate(file: UploadFile = File(...)):
    content = await file.read()

    # BOM付きUTF-8 / Shift-JIS 両対応
    for enc in ("utf-8-sig", "shift_jis", "utf-8"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise HTTPException(status_code=400, detail="CSVのエンコーディングを認識できませんでした")

    df = pd.read_csv(io.StringIO(text))

    # マネーフォワードの保有証券CSVカラムを検索
    col_map = {}
    for col in df.columns:
        col_s = col.strip()
        if "銘柄コード" in col_s or "コード" in col_s:
            col_map["code"] = col
        elif "銘柄名" in col_s or "銘柄" in col_s:
            col_map["name"] = col
        elif "評価額" in col_s:
            col_map["value"] = col
        elif "評価損益" not in col_s and ("現在値" in col_s or "時価" in col_s):
            col_map.setdefault("price", col)
        elif "保有金融機関" in col_s or "口座" in col_s:
            col_map["account"] = col

    if "value" not in col_map or ("code" not in col_map and "name" not in col_map):
        raise HTTPException(
            status_code=400,
            detail=f"必要なカラム（銘柄名またはコード・評価額）が見つかりません。検出されたカラム: {list(df.columns)}",
        )

    items: list[HoldingItem] = []

    for _, row in df.iterrows():
        raw_code = str(row.get(col_map.get("code", ""), "")).strip()
        if raw_code.lower() == "nan" or raw_code == "-":
            raw_code = ""
        name = str(row.get(col_map.get("name", ""), raw_code)).strip()
        if name.lower() == "nan":
            name = raw_code
        if not raw_code and not name:
            continue

        # 評価額: カンマ除去して数値変換
        raw_value = str(row[col_map["value"]]).replace(",", "").replace("¥", "").strip()
        try:
            value = float(raw_value)
        except ValueError:
            continue

        account = str(row.get(col_map.get("account", ""), "")).strip()
        items.append(HoldingItem(name=name, code=raw_code, value=value, account=account))

    return process_holdings(items)


class HoldingItem(BaseModel):
    name: str
    code: str = ""
    value: float
    account: str = ""
    quantity: float | None = None


class HoldingsPayload(BaseModel):
    holdings: list[HoldingItem]


def process_holdings(items: list[HoldingItem]) -> dict:
    # 各保有行を先にNISA判定し、その後に同一銘柄を合算する。
    # accountが渡された場合は口座名を優先し、旧拡張機能ではnisa_codes.txtの
    # 記載件数分だけ先頭の一致行をNISAとして消費する。
    nisa_lots = load_nisa_lot_counts()
    aliases = load_ticker_aliases()
    yield_overrides = load_yield_overrides()
    merged: dict[str, dict] = {}

    for item in items:
        if item.value <= 0:
            continue
        code, ticker = resolve_ticker(item.code, item.name, aliases)
        nisa_key = _normalize_nisa_code(code) if code else _normalize_config_key(item.name)
        has_explicit_account = bool(item.account.strip())
        if has_explicit_account:
            nisa = "NISA" in item.account.upper()
        else:
            nisa = bool(nisa_key and nisa_lots[nisa_key] > 0)
            if nisa:
                nisa_lots[nisa_key] -= 1

        key = ticker or _normalize_config_key(item.name)
        group = merged.setdefault(key, {
            "code": code,
            "ticker": ticker,
            "name": item.name,
            "lots": [],
        })
        if not group["code"] and code:
            group["code"] = code
        if item.name and item.name != item.code:
            group["name"] = item.name
        group["lots"].append({"value": item.value, "nisa": nisa})

    results = []
    total_value = 0.0
    total_monthly = 0.0
    total_monthly_net = 0.0
    covered_value = 0.0

    for i, group in enumerate(merged.values()):
        ticker = group["ticker"]
        code = group["code"]
        value = sum(lot["value"] for lot in group["lots"])

        # リクエスト間に0.3秒のインターバルを設けてレートリミットを回避
        if i > 0:
            time.sleep(0.3)

        ticker_data = fetch_ticker_data(ticker) if ticker else {}
        name = get_display_name(code, group["name"], ticker_data.get("name"))
        div_yield, yield_source = get_dividend_yield(
            code, ticker, name, ticker_data.get("div_yield"), yield_overrides
        )

        tax_category = "JP" if not ticker or ticker.endswith(".T") else "US"
        dividend_checked_at = datetime.now(timezone.utc).isoformat()
        if div_yield is not None:
            monthly_exact = value * div_yield / 12
            monthly_net_exact = 0.0
            for lot in group["lots"]:
                if lot["nisa"]:
                    rate = nisa_tax_rate_for(ticker) if ticker else 0.0
                else:
                    rate = tax_rate_for(ticker) if ticker else TAX_JP
                monthly_net_exact += lot["value"] * div_yield / 12 * (1 - rate)
            monthly = round(monthly_exact, 0)
            monthly_net = round(monthly_net_exact, 0)
            effective_tax = 1 - (monthly_net_exact / monthly_exact) if monthly_exact else 0.0
            covered_value += value
        else:
            monthly = None
            monthly_net = None
            effective_tax = 0.0

        nisa_count = sum(1 for lot in group["lots"] if lot["nisa"])
        if nisa_count == 0:
            nisa_status = "none"
        elif nisa_count == len(group["lots"]):
            nisa_status = "all"
        else:
            nisa_status = "partial"

        total_value += value
        if monthly is not None:
            total_monthly += monthly_exact
            total_monthly_net += monthly_net_exact

        results.append({
            "code": code,
            "ticker": ticker,
            "name": name,
            "value": value,
            "div_yield": round(div_yield * 100, 2) if div_yield is not None else None,
            "yield_source": yield_source,
            "tax_category": tax_category,
            "dividend_checked_at": dividend_checked_at,
            "annual_dividend": round(monthly_exact * 12) if monthly is not None else None,
            "annual_dividend_net": round(monthly_net_exact * 12) if monthly_net is not None else None,
            "tax_rate": round(effective_tax * 100, 3),
            "nisa": nisa_status == "all",
            "nisa_status": nisa_status,
            "nisa_value": sum(lot["value"] for lot in group["lots"] if lot["nisa"]),
            "monthly_dividend": monthly,
            "monthly_dividend_net": monthly_net,
        })

    annual_dividend = round(total_monthly * 12, 0)
    fx_rate = fetch_usd_jpy_rate()
    return {
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "holdings_as_of": None,
        "missing_yield_count": sum(r["div_yield"] is None for r in results),
        "fx_fetched_at": datetime.fromtimestamp(_fx_cache["ts"], timezone.utc).isoformat() if _fx_cache["ts"] else None,
        "holdings": results,
        "total_value": total_value,
        "total_monthly_dividend": round(total_monthly, 0),
        "total_annual_dividend": annual_dividend,
        "total_monthly_dividend_net": round(total_monthly_net, 0),
        "total_annual_dividend_net": round(total_monthly_net * 12, 0),
        "portfolio_dividend_yield": round(annual_dividend / total_value * 100, 2) if total_value else None,
        "yield_coverage_value": round(covered_value, 0),
        "yield_coverage_pct": round(covered_value / total_value * 100, 1) if total_value else 0.0,
        "usd_jpy_rate": fx_rate,
    }


@app.post("/api/calculate-from-json")
async def calculate_from_json(payload: HoldingsPayload):
    if not payload.holdings:
        raise HTTPException(status_code=400, detail="保有銘柄データが空です")
    result = process_holdings(payload.holdings)
    sid = uuid.uuid4().hex[:10]
    _sessions[sid] = result
    result["session_id"] = sid
    return result


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="セッションが見つかりません（サーバーを再起動した可能性があります）")
    return _sessions[session_id]
