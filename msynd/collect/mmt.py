from datetime import date
import json
import re

from msynd.collect.dispatcher import Dispatcher, FetchResult, real_transport
from msynd.collect.railradar import validate_request
from msynd.config import settings
from msynd.db import connect

HOST = "rails-ris.makemytrip.com"


def not_running(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return False
    return (isinstance(payload, dict) and payload.get('success') is False
            and isinstance(payload.get('error'), dict)
            and payload['error'].get('errorCode') in (1001, 1002))


def breaker_exempt(response) -> bool:
    return response.status_code == 400 and not_running(response.content)


def classify(purpose, response) -> str:
    body = response.content.lower()
    if (response.status_code == 403 or "text/html" in response.headers.get("Content-Type", "").lower()
            or re.search(br"<(?:!doctype\s+html|html\b)", body) or b"captcha" in body):
        return "challenge"
    if purpose != "status" or not 200 <= response.status_code < 300:
        return "invalid"
    try:
        json.loads(response.content)
    except ValueError:
        return "invalid"
    # ponytail: JSON validity only; add semantic run validation with the Task 10 parser.
    return "ok"


class MmtClient:
    def __init__(self, d: Dispatcher):
        if d.host != HOST:
            raise ValueError("MMT client requires its own host dispatcher")
        if d.classify is not None and d.classify is not classify:
            raise ValueError("MMT dispatcher requires its status classifier")
        self.d = d
        self.d.classify = classify
        self.d.breaker_exempt = breaker_exempt

    @classmethod
    def from_env(cls, *, budgets=None):
        transport = real_transport()
        return cls(Dispatcher(HOST, settings.mmt_rate, connect, transport, classify=classify, budgets=budgets))

    async def train_running(self, train: str, start: date, run_id=None, *, mode='poll', max_attempts=5,
                            stop_after_5xx: int | None = None) -> FetchResult:
        validate_request(train, start)
        return await self.d.request("status", "POST", f"https://{HOST}/api/ris/train/livestatus/v2",
            key=f"{train}/{start.isoformat()}", run_id=run_id, mode=mode, max_attempts=max_attempts,
            stop_after_5xx=stop_after_5xx,
            json={"trainNumber": train, "dateOfJourney": start.strftime("%d-%m-%Y"), "findNextRunningDate": False},
            headers={"Content-Type": "application/json"})
