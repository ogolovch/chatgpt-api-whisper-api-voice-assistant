## pip install google-cloud-speech
# This code works on Windows 10
# The provided code uses Gradio for creating a user interface, 
# Google Cloud Speech-to-Text for transcribing audio input, and 
# OpenAI GPT-3.5 Turbo for generating responses based on the transcribed text. 
# It also includes a function to use pyttsx3 for converting the generated text response to speech.

import gradio as gr
import openai, config, subprocess
import pyttsx3
import warnings
import requests  
from google.cloud import speech_v1p1beta1 as speech
import io
import os
import json

warnings.filterwarnings("ignore", category=UserWarning, module="gradio.processing_utils")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/your/path/to/google_credentials.json"
openai.api_key = config.OPENAI_API_KEY

messages = [{"role": "system", "content": 'You are a Python coach. Respond to all input in 250 words or less.'}]

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def transcribe(audio):
    global messages

    client = speech.SpeechClient()

    with io.open(audio, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=48000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        transcript = result.alternatives[0].transcript

    messages.append({"role": "user", "content": transcript})

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
    }

    response = requests.post(url, headers=headers, json=data).json()


    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    speak(system_message['content'])

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text")
ui.launch()

