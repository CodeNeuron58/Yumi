from langgraph.graph import StateGraph, END
from typing import TypedDict
from nodes import chat_node

class AgentState(TypedDict):
    input: str
    response: str


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("chat", chat_node)

    graph.set_entry_point("chat")

    graph.add_edge("chat", END)

    return graph.compile()