import os
from dotenv import load_dotenv
import io
import wave
import numpy as np
import sounddevice as sd

from typecast import Typecast
from typecast.models import TTSRequest, SmartPrompt

class YumiSpeaker:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("TYPECAST_API_KEY")
        if not api_key:
            raise ValueError("TYPECAST_API_KEY not found in environment variables.")
        
        # Initialize client
        self.client = Typecast(api_key=api_key)
        
        # Constants for requested voice
        self.model = "ssfm-v21"
        self.voice_id = "tc_6359e7f6467f9e240b68292c"

    def speak(self, text: str):
        if not text:
            return

        print(f"Synthesizing voice...")
        try:
            # Convert text to speech
            response = self.client.text_to_speech(TTSRequest(
                text=text,
                model=self.model,
                voice_id=self.voice_id,
                prompt=SmartPrompt(
                    emotion_type="smart"
                )
            ))
            
            # Read from audio_data bytes into wave
            audio_data = io.BytesIO(response.audio_data)
            with wave.open(audio_data, 'rb') as wf:
                # Get audio parameters
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                
                # Determine correct numpy dtype based on sample width
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                    
                # Read all frames into numpy array
                frames = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(frames, dtype=dtype)
                
                # Reshape for multi-channel if necessary
                if channels > 1:
                    # In wave files, channels are interleaved
                    audio_array = audio_array.reshape(-1, channels)
            
            # Play audio using sounddevice
            # blocking play
            sd.play(audio_array, samplerate=framerate)
            sd.wait()
            
        except Exception as e:
            print(f"Error speaking response: {e}")
