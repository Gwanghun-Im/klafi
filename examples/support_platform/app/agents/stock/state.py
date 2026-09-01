from typing import TypedDict


class StockState(TypedDict):
    symbol: str
    quantity: int
    result: str
    fill: dict
