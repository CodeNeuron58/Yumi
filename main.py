from graph import build_graph

graph = build_graph()

while True:

    user_input = input("You: ")

    result = graph.invoke({
        "input": user_input
    })

    print("AI:", result["response"].content)