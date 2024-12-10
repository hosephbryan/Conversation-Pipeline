# import time
import numpy as np
import torch
import speech_recognition as sr
import keyboard  # For key press detection
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch.nn.functional as F
from SentimentAnalysis import analyze_text_emotions  # Import the emotion analysis function

# Initialize the recognizer for SpeechRecognition
def detect__voice_emotions(emotion_model, emotion_processor, emotion_labels, audio_data, sample_rate=16000):
    inputs = emotion_processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        emotion_logits = emotion_model(inputs.input_values).logits
    probs = F.softmax(emotion_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 3)
    return [(emotion_labels[i], top_probs[0][idx].item()) for idx, i in enumerate(top_indices[0])]

async def listen_for_commands():
    emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-large-superb-er")
    emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-er")
    # Emotion labels (adjust as needed based on the model's labels)
    emotion_labels = ["neutral", "happy", "angry", "sad", "fear", "surprise", "disgust"]

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Press and hold the Spacebar to speak. Release to stop.")
        final_output = None
        while final_output is None:
            if keyboard.is_pressed("space"):
                print("Listening for command...")
                try:
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=12)
                    command_text = recognizer.recognize_google(audio)
                    command_string = str(command_text)
                    # Convert audio to float for voice emotion detection
                    raw_audio_data = audio.get_raw_data()
                    audio_data_np = np.frombuffer(raw_audio_data, np.int16).astype(np.float32) / 32768.0
                    voice_emotions = detect__voice_emotions(emotion_model, emotion_processor, emotion_labels, audio_data_np)
                    voice_emotion_output = ", ".join([f"{label} ({prob:.2f})" for label, prob in voice_emotions])
                        
                    # Analyze text-based emotions
                    text_emotions = analyze_text_emotions(command_text)
                    text_emotion_output = ", ".join([f"{label} ({score:.2f})" for label, score in text_emotions])

                    # Final formatted output
                    final_output = (
                        f"{command_text}. "
                        f"The person's speech patterns seem to indicate {voice_emotion_output}, "
                        f"and the sentiment behind the person's text is {text_emotion_output}."
                    )
                        
                    # Print and store the final output
                    print("Final Output:")
                    print(final_output)
                    return final_output, command_string
                
                except sr.UnknownValueError:
                    print("Could not understand audio.")
                except sr.RequestError as e:
                    print(f"Speech Recognition error; {e}")
                except Exception as e:
                    print(f"An error occurred: {e}")

async def detect_emotion():
    emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-large-superb-er")
    emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-er")
    # Emotion labels (adjust as needed based on the model's labels)
    emotion_labels = ["neutral", "happy", "angry", "sad", "fear", "surprise", "disgust"]
    detected_emotion = None
    stt_output = None

    while detected_emotion is None:
        detected_emotion, stt_output = await listen_for_commands(emotion_processor, emotion_model, emotion_labels)

    return detected_emotion, stt_output