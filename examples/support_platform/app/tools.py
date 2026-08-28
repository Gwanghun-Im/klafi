"""공통 Tool — 프로젝트의 모든 Tool을 여기서 관리한다.

Tool은 "무엇을 할 수 있는가"만 담는다(권한·입력검증·Timeout·Metric).
"언제 쓰는가"(사용 지침)는 Tool이 아니라 Skill의 몫 — skills.py 참고.
"""

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from klafi.tool import tool


# ── 주문 ────────────────────────────────────────────────────────────────
class LookupIn(BaseModel):
    order_id: str


@tool(name="lookup_order", description="주문 상태 조회", required_permission="orders:read", input_schema=LookupIn)
def lookup_order(order_id: str) -> dict:
    # 실제로는 DB/외부 API 조회
    return {"order_id": order_id, "status": "배송중", "eta": "2일 내"}


# ── 사내 규정 ───────────────────────────────────────────────────────────
class PolicyIn(BaseModel):
    keyword: str


_POLICY = {
    "환불": "구매 후 7일 이내 미개봉 상품에 한해 전액 환불.",
    "교환": "동일 상품에 한해 14일 이내 1회 교환 가능.",
    "배송": "주문 후 영업일 기준 2~3일 소요, 도서산간 추가 1일.",
}


@tool(name="search_policy", description="사내 규정 조회", required_permission="policy:read", input_schema=PolicyIn)
def search_policy(keyword: str) -> dict:
    hit = next((v for k, v in _POLICY.items() if k in keyword), "해당 규정을 찾지 못했습니다.")
    return {"keyword": keyword, "policy": hit}


# ── 시각 ────────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))


@tool(name="kst_now", description="현재 한국(KST) 날짜와 시각")
def kst_now() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


# ── 주식 매수 (MOCK — 실제 거래 아님, 데모용) ────────────────────────────
class BuyStockIn(BaseModel):
    symbol: str
    quantity: int


_PRICES = {"AAPL": 230, "TSLA": 250, "NVDA": 175, "005930": 71000}  # mock 시세


@tool(name="get_quote", description="주식 현재가 조회(mock)")
def get_quote(symbol: str) -> dict:
    price = _PRICES.get(symbol.upper() if symbol.isalpha() else symbol, 100)
    return {"symbol": symbol, "price": price}


@tool(
    name="buy_stock",
    description="주식 매수 주문 실행(mock). 실제 체결이 아니라 모의 체결 결과를 돌려준다.",
    required_permission="trades:write",   # 최소권한 — 매수는 명시 권한 필요
    input_schema=BuyStockIn,
)
def buy_stock(symbol: str, quantity: int) -> dict:
    # 실제로는 증권사 API 호출. 여기서는 모의 체결.
    import uuid

    price = _PRICES.get(symbol.upper() if symbol.isalpha() else symbol, 100)
    return {
        "order_id": uuid.uuid4().hex[:8],
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "amount": price * quantity,
        "status": "FILLED",
    }
