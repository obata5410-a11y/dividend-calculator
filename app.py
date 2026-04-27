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
import re
import uuid
import time

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


def fetch_ticker_data(ticker: str) -> dict:
    """日本株（4桁.T）はminkabu、それ以外はyfinanceを使う。"""
    m = re.fullmatch(r"(\d{4})\.T", ticker)
    if m:
        return _fetch_jp_ticker_data(m.group(1))
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


def tax_rate_for(ticker: str) -> float:
    """ティッカーに応じた源泉徴収税率を返す。"""
    return TAX_JP if ticker.endswith(".T") else TAX_US


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

    if "code" not in col_map or "value" not in col_map:
        raise HTTPException(
            status_code=400,
            detail=f"必要なカラム（銘柄コード・評価額）が見つかりません。検出されたカラム: {list(df.columns)}",
        )

    results = []
    total_value = 0.0
    total_monthly = 0.0

    for _, row in df.iterrows():
        raw_code = str(row[col_map["code"]]).strip()
        if not raw_code or raw_code in ("nan", "-", ""):
            continue

        name = str(row.get(col_map.get("name", ""), raw_code)).strip()

        # 評価額: カンマ除去して数値変換
        raw_value = str(row[col_map["value"]]).replace(",", "").replace("¥", "").strip()
        try:
            value = float(raw_value)
        except ValueError:
            continue

        ticker = normalize_ticker(raw_code)
        if results:
            time.sleep(0.3)
        td = fetch_ticker_data(ticker)
        div_yield = td.get("div_yield")
        if td.get("name") and name == raw_code:
            name = td["name"]
        monthly = round(value * div_yield / 12, 0) if div_yield else None
        tax = tax_rate_for(ticker)
        monthly_net = calc_after_tax(monthly, tax)

        total_value += value
        if monthly:
            total_monthly += monthly

        results.append(
            {
                "code": raw_code,
                "ticker": ticker,
                "name": name,
                "value": value,
                "div_yield": round(div_yield * 100, 2) if div_yield else None,
                "tax_rate": round(tax * 100, 3),
                "monthly_dividend": monthly,
                "monthly_dividend_net": monthly_net,
            }
        )

    total_monthly_net = sum(
        r["monthly_dividend_net"] for r in results if r["monthly_dividend_net"] is not None
    )
    return {
        "holdings": results,
        "total_value": total_value,
        "total_monthly_dividend": round(total_monthly, 0),
        "total_annual_dividend": round(total_monthly * 12, 0),
        "total_monthly_dividend_net": round(total_monthly_net, 0),
        "total_annual_dividend_net": round(total_monthly_net * 12, 0),
    }


class HoldingItem(BaseModel):
    name: str
    code: str = ""
    value: float


class HoldingsPayload(BaseModel):
    holdings: list[HoldingItem]


def process_holdings(items: list[HoldingItem]) -> dict:
    # 同一コードを複数証券会社分で合算する
    merged: dict[str, HoldingItem] = {}
    for item in items:
        if item.value <= 0:
            continue
        key = item.code if item.code else item.name
        if key in merged:
            merged[key] = HoldingItem(
                name=item.name if item.name != item.code else merged[key].name,
                code=item.code,
                value=merged[key].value + item.value,
            )
        else:
            merged[key] = item

    results = []
    total_value = 0.0
    total_monthly = 0.0

    for i, item in enumerate(merged.values()):
        ticker = normalize_ticker(item.code) if item.code else ""

        # リクエスト間に0.3秒のインターバルを設けてレートリミットを回避
        if i > 0:
            time.sleep(0.3)

        ticker_data = fetch_ticker_data(ticker) if ticker else {}
        div_yield = ticker_data.get("div_yield")
        name = get_display_name(item.code, item.name, ticker_data.get("name"))
        monthly = round(item.value * div_yield / 12, 0) if div_yield else None
        tax = tax_rate_for(ticker) if ticker else TAX_JP
        monthly_net = calc_after_tax(monthly, tax)

        total_value += item.value
        if monthly:
            total_monthly += monthly

        results.append({
            "code": item.code,
            "ticker": ticker,
            "name": name,
            "value": item.value,
            "div_yield": round(div_yield * 100, 2) if div_yield else None,
            "tax_rate": round(tax * 100, 3),
            "monthly_dividend": monthly,
            "monthly_dividend_net": monthly_net,
        })

    total_monthly_net = sum(
        r["monthly_dividend_net"] for r in results if r["monthly_dividend_net"] is not None
    )
    return {
        "holdings": results,
        "total_value": total_value,
        "total_monthly_dividend": round(total_monthly, 0),
        "total_annual_dividend": round(total_monthly * 12, 0),
        "total_monthly_dividend_net": round(total_monthly_net, 0),
        "total_annual_dividend_net": round(total_monthly_net * 12, 0),
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
