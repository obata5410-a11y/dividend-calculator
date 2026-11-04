"""Private, append-only Notion history store. Tokens never leave the server."""
import hashlib
import json
import os
import secrets
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Literal

import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

router = APIRouter(prefix="/api/history", tags=["history"])
_JST = timezone(timedelta(hours=9))
_write_lock = threading.Lock()
_api_lock = threading.Lock()
_last_request = 0.0


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    date: date
    scope: Literal["simulator", "mf_securities"]
    total_value: float = Field(ge=0)
    annual_net: float | None = Field(default=None, ge=0)
    annual_gross: float | None = Field(default=None, ge=0)
    coverage: float | None = Field(default=None, ge=0, le=100)
    source: str = Field(default="保存記録", max_length=200)
    method: str = Field(default="unknown", max_length=80)
    calculated_at: datetime | None = None

    @field_validator("date")
    @classmethod
    def not_future(cls, value):
        if value > datetime.now(_JST).date():
            raise ValueError("未来日の履歴は登録できません")
        return value

    @field_validator("calculated_at")
    @classmethod
    def aware_time(cls, value):
        if value is not None:
            if value.tzinfo is None:
                raise ValueError("計算日時にはタイムゾーンが必要です")
            if value > datetime.now(timezone.utc) + timedelta(minutes=5):
                raise ValueError("未来の計算日時は登録できません")
        return value


def snapshot_key(record: Snapshot) -> str:
    # Source wording is not identity. Re-importing identical figures is idempotent.
    data = record.model_dump(mode="json", exclude={"source"})
    wire = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(wire.encode()).hexdigest()


def configured() -> bool:
    return bool(os.getenv("NOTION_TOKEN") and os.getenv("NOTION_DATA_SOURCE_ID")
                and len(os.getenv("DIVIDEND_HISTORY_ACCESS_KEY", "")) >= 32)


def authorize(authorization: str | None = Header(default=None)):
    expected = os.getenv("DIVIDEND_HISTORY_ACCESS_KEY", "")
    supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    if len(expected) < 32 or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="履歴用アクセスキーを確認してください")
    if not configured():
        raise HTTPException(status_code=503, detail="サーバーのNotion接続設定が未完了です")


class NotionStore:
    def __init__(self):
        self.data_source_id = os.environ["NOTION_DATA_SOURCE_ID"]
        self.headers = {"Authorization": "Bearer " + os.environ["NOTION_TOKEN"],
                        "Notion-Version": "2025-09-03", "Content-Type": "application/json"}

    def request(self, path, payload):
        # Never automatically retry writes; a timeout can occur AFTER Notion committed.
        global _last_request
        try:
            with _api_lock:
                delay = max(0, 0.35 - (time.monotonic() - _last_request))
                if delay:
                    time.sleep(delay)
                _last_request = time.monotonic()
            r = requests.post("https://api.notion.com/v1/" + path,
                              json=payload, headers=self.headers, timeout=(5, 25))
        except requests.RequestException as exc:
            raise HTTPException(status_code=504, detail="Notionの応答を確認できません。再読込して登録状態を確認してください") from exc
        if not r.ok:
            raise HTTPException(status_code=502, detail="Notionへの接続に失敗しました。接続権限・データソース設定を確認してください")
        return r.json()

    def pages(self, filter_=None):
        payload = {"page_size": 100}
        if filter_:
            payload["filter"] = filter_
        rows = []
        while True:
            d = self.request(f"data_sources/{self.data_source_id}/query", payload)
            rows.extend(d["results"])
            if not d.get("has_more"):
                return rows
            cursor = d.get("next_cursor")
            if not cursor or cursor == payload.get("start_cursor") or len(rows) >= 10000:
                raise HTTPException(status_code=502, detail="Notion履歴を最後まで取得できませんでした")
            payload["start_cursor"] = cursor

    @staticmethod
    def properties(r: Snapshot):
        rich = lambda text: {"rich_text": [{"text": {"content": text}}]}
        p = {"Name": {"title": [{"text": {"content": f"{r.date.isoformat()} | {r.scope}"}}]},
             "SnapshotKey": rich(snapshot_key(r)), "Date": {"date": {"start": r.date.isoformat()}},
             "Scope": {"select": {"name": r.scope}}, "TotalValue": {"number": r.total_value},
             "AnnualNet": {"number": r.annual_net}, "AnnualGross": {"number": r.annual_gross},
             "Coverage": {"number": r.coverage}, "Source": rich(r.source), "Method": rich(r.method)}
        if r.calculated_at:
            p["CalculatedAt"] = {"date": {"start": r.calculated_at.isoformat()}}
        return p

    def append(self, record):
        with _write_lock:
            matches = self.pages({"property": "SnapshotKey", "rich_text": {"equals": snapshot_key(record)}})
            if matches:
                return False
            self.request("pages", {"parent": {"type": "data_source_id", "data_source_id": self.data_source_id},
                                   "properties": self.properties(record)})
            return True

    def records(self):
        # Fetch in full before displaying; do not silently return partial financial history.
        latest = {}
        for page in self.pages():
            p = page["properties"]
            text = lambda key: "".join(t.get("plain_text", t.get("text", {}).get("content", "")) for t in p[key].get("rich_text", []))
            try:
                record = Snapshot(date=p["Date"]["date"]["start"], scope=p["Scope"]["select"]["name"],
                    total_value=p["TotalValue"]["number"], annual_net=p["AnnualNet"]["number"],
                    annual_gross=p["AnnualGross"]["number"], coverage=p["Coverage"]["number"],
                    source=text("Source"), method=text("Method"),
                    calculated_at=(p.get("CalculatedAt", {}).get("date") or {}).get("start"))
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=502, detail="Notion履歴に形式の異なる行があります。台帳を確認してください") from exc
            # New calculations win over late re-imports of older calculations.
            stamp = record.calculated_at or datetime.fromisoformat(page["created_time"].replace("Z", "+00:00"))
            key = (record.scope, record.date)
            if key not in latest or stamp > latest[key][0]:
                latest[key] = (stamp, record)
        return [item[1].model_dump(mode="json") for _, item in sorted(latest.items(), key=lambda pair: (pair[0][1], pair[0][0]))]


class SnapshotBatch(BaseModel):
    records: list[Snapshot] = Field(min_length=1, max_length=25)


@router.get("/status")
def status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return {"configured": configured()}


@router.get("", dependencies=[Depends(authorize)])
def read_history(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return {"records": NotionStore().records(), "storage": "notion"}


@router.post("", dependencies=[Depends(authorize)])
def append_history(batch: SnapshotBatch, response: Response):
    response.headers["Cache-Control"] = "no-store"
    store = NotionStore()
    created = sum(store.append(r) for r in batch.records)
    return {"created": created, "unchanged": len(batch.records) - created, "storage": "notion"}
