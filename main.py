import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import TypedDict
from langgraph.graph import StateGraph, END

from Yumi_Hears.pipeline import AudioPipeline
from Yumi_Speaks.tts import YumiSpeaker
from Yumi_Brain.nodes import chat_node

class MainState(TypedDict):
    input: str
    response: str
    session_id: str

def main():
    print("Initializing Yumi Audio Pipeline...")
    pipeline = AudioPipeline()
    
    print("Initializing Yumi Speaker (TTS)...")
    speaker = YumiSpeaker()
    
    session_id = "yumi_session_1"
    
    # --- Define Graph Nodes ---
    def listen_node(state: MainState):
        text = pipeline.run_cycle()
        return {"input": text}
        
    def think_node(state: MainState):
        print(f"User (Audio): {state['input']}")
        # Utilize existing brain node which wraps conversational memory and the LLM
        result = chat_node({
            "input": state["input"],
            "session_id": state["session_id"]
        })
        return {"response": result["response"]}
        
    def speak_node(state: MainState):
        response_text = state["response"]
        print(f"Yumi: {response_text}")
        speaker.speak(response_text)
        print("-" * 50)
        return {"response": response_text}
        
    # --- Define Routing ---
    def should_think(state: MainState):
        # Only proceed to thinking/speaking if audio was transcribed successfully
        if state.get("input") and state["input"].strip():
            return "think"
        return "end"

    print("Building LangGraph...")
    workflow = StateGraph(MainState)
    
    # Add nodes
    workflow.add_node("listen", listen_node)
    workflow.add_node("think", think_node)
    workflow.add_node("speak", speak_node)
    
    # Set entry point
    workflow.set_entry_point("listen")
    
    # Add edges
    workflow.add_conditional_edges(
        "listen",
        should_think,
        {
            "think": "think",
            "end": END
        }
    )
    workflow.add_edge("think", "speak")
    workflow.add_edge("speak", END)
    
    # Compile the graph
    app = workflow.compile()
    
    print("System ready. Listening for speech...")
    while True:
        try:
            # We invoke the graph to handle one full turn of conversation.
            # Doing this inside a while loop avoids LangGraph `recursion_limit` issues 
            # for endless background listening.
            app.invoke({
                "input": "", 
                "response": "", 
                "session_id": session_id
            })
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()