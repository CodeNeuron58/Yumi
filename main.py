import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Yumi_Brain.graph import build_graph
from Yumi_Hears.pipeline import AudioPipeline
from Yumi_Speaks.tts import YumiSpeaker

def main():
    print("Initializing Yumi Audio Pipeline...")
    pipeline = AudioPipeline()
    
    print("Initializing Yumi Brain (LLM)...")
    graph = build_graph()
    
    print("Initializing Yumi Speaker (TTS)...")
    speaker = YumiSpeaker()
    
    session_id = "yumi_session_1"
    
    print("System ready. Listening for speech...")
    while True:
        try:
            # Captures and transcribes audio
            text = pipeline.run_cycle()
            
            # If transcription is successful and not empty
            if text:
                print(f"User (Audio): {text}")
                
                # Send context to graph
                result = graph.invoke({
                    "input": text,
                    "session_id": session_id
                })
                
                response_text = result['response']
                print(f"Yumi: {response_text}")
                
                # Speak the response out loud
                speaker.speak(response_text)
                
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()