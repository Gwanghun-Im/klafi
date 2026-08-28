"""업무개발자 영역 — 주식 매수 Agent (HITL 승인 게이트).

    quote → (사람 승인 대기) → execute

매수는 되돌리기 어려운 행동이라 **실행 전 사람 승인**을 받는다. `request_approval`이 시세·수량·
예상금액을 담아 interrupt로 중단하고, 승인/반려로 재개(resume)하면 그 결정에 따라 체결하거나 취소한다.
체크포인터가 중단 상태를 보관하므로 재시작·다른 프로세스에서도 이어서 승인할 수 있다.

주의: buy_stock은 mock(모의 체결)이다. 실제 거래 API를 붙이는 자리에 그대로 대체하면 된다.
"""

from typing import TypedDict

from langgraph.graph import END, START

from klafi.core import AgentSpec, KlafiGraph, klafi_node
from klafi.hitl import request_approval

from ..tools import buy_stock, get_quote


class StockState(TypedDict):
    symbol: str
    quantity: int
    result: str
    fill: dict


class StockAgent(KlafiGraph):
    spec = AgentSpec(id="stock", name="Stock Trading Agent", version="1.0.0", agent_type="hitl", model="fast")
    state_schema = StockState

    def define(self):
        @klafi_node("trade")
        def trade(state: StockState) -> dict:
            symbol, qty = state["symbol"], state["quantity"]
            quote = get_quote.run(symbol=symbol)  # mock 시세
            est = quote["price"] * qty

            # ── HITL: 매수 실행 전 사람 승인 (여기서 interrupt) ──
            decision = request_approval(
                "주식 매수 승인",
                payload={"symbol": symbol, "quantity": qty, "price": quote["price"], "est_amount": est},
                approver="trader",
            )
            if not decision.approved:
                return {"result": f"매수 취소됨: {symbol} x{qty} ({decision.comment or '반려'})"}

            fill = buy_stock.run(symbol=symbol, quantity=qty)  # 승인 후에만 체결
            return {"result": f"{symbol} {qty}주 체결(주문 {fill['order_id']}, {fill['amount']:,}원)", "fill": fill}

        self.add_node("trade", trade)
        self.add_edge(START, "trade")
        self.add_edge("trade", END)
