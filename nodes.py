from llm import get_llm

llm = get_llm()

def chat_node(state):

    user_input = state["input"]

    response = llm.invoke(user_input)

    return {"response": response}