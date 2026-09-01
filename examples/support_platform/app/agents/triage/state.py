from langgraph.graph import MessagesState


class TriageState(MessagesState):
    route: str  # "simple" | "complex"
