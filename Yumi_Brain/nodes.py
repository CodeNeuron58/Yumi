from langchain_core.runnables.history import RunnableWithMessageHistory
from Yumi_Brain.memory.chat_history import get_session_history
from Yumi_Brain.llm import chain

chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def chat_node(state):
    session_id = state["session_id"]
    user_input = state["input"]

    # We now expect a Pydantic YumiResponse object instead of an AIMessage (due to structured output)
    response_obj = chat_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    # Return the dictionary representing all dimensions of the LLM state output
    return {
        "response": response_obj.response_text,
        "expression": response_obj.expression,
        "motion": response_obj.motion
    }