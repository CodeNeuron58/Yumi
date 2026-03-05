from langchain_core.runnables.history import RunnableWithMessageHistory
from memory.chat_history import get_session_history
from llm import chain

chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def chat_node(state):

    session_id = state["session_id"]
    user_input = state["input"]

    response = chat_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    return {"response": response.content}