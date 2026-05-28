import asyncio
import logging
import os
import json
from typing import Dict, Any, List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from rich.console import Console
from rich.panel import Panel
from google import genai  # 使用新版 SDK
import edge_tts
import pygame

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Rich console for UI
console = Console()

# 1. Load environment variables and initialize Gemini Client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    # 新版初始化方式：实例化一个全局 client 对象
    client = genai.Client(api_key=api_key)
else:
    client = None
    logger.warning("GEMINI_API_KEY not found in environment variables.")

async def record_audio() -> str:
    """
    Listen for the Enter key to start and stop audio recording.
    Records audio using sounddevice and saves it to a temporary .wav file.
    """
    fs = 44100
    channels = 1
    recording = []

    def callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")
        recording.append(indata.copy())

    console.print("\n[bold cyan]Press Enter to start recording...[/bold cyan]")
    await asyncio.to_thread(input)
    
    console.print("[bold red]Recording... Press Enter to stop.[/bold red]")
    stream = sd.InputStream(samplerate=fs, channels=channels, callback=callback)
    with stream:
        await asyncio.to_thread(input)
    
    console.print("[bold green]Recording stopped.[/bold green]")
    
    audio_data = np.concatenate(recording, axis=0)
    file_path = "temp.wav"
    wavfile.write(file_path, fs, audio_data)
    return file_path

class InterviewResponse(BaseModel):
    corrected_sentence: str
    grammar_mistakes: List[str]
    fluency_score: float
    next_question: str

async def chat_with_gemini(audio_path: str) -> Dict[str, Any]:
    """
    Upload the .wav file directly to the Gemini Multimodal API.
    Bundles the audio with the Interviewer System Prompt and enforces
    a structured JSON response containing grammar corrections and the next question.
    """
    if not client:
        raise ValueError("Gemini client is not initialized. Please check your GEMINI_API_KEY.")

    console.print("[cyan]Uploading audio and querying Gemini...[/cyan]")
    
    # 【改动 1】新版文件上传语法：使用 client.files.upload
    audio_file = await asyncio.to_thread(client.files.upload, file=audio_path)
    
    prompt = (
        "You are an English technical interviewer. Evaluate the user's spoken response. "
        "Return a JSON object with exactly these keys: "
        "'corrected_sentence' (string, the corrected version of what they said), "
        "'grammar_mistakes' (list of strings, detailing errors), "
        "'fluency_score' (float, 1 to 10), "
        "'next_question' (string, a relevant follow-up technical interview question)."
    )
    
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-3.5-flash",
            contents=[prompt, audio_file],
            config={
                "response_mime_type": "application/json",
                "response_schema": InterviewResponse,
            }
        )
        
        result = json.loads(response.text)
        return result
    finally:
        await asyncio.to_thread(client.files.delete, name=audio_file.name)

async def play_audio(text: str) -> None:
    """
    Synthesize and stream-play the provided text using edge-tts.
    """
    if not text:
        return

    console.print("[cyan]Synthesizing speech...[/cyan]")
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    audio_file = "response.mp3"
    await communicate.save(audio_file)
    
    console.print("[cyan]Playing audio...[/cyan]")
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)
        
    pygame.mixer.quit()
    
    if os.path.exists(audio_file):
        os.remove(audio_file)

def render_ui(response_data: Dict[str, Any]) -> None:
    """
    Use the rich library to format and print technical grammar corrections beautifully.
    """
    console.print("\n")
    console.print(Panel(response_data.get("corrected_sentence", "N/A"), title="[bold green]Corrected Sentence[/bold green]", expand=False))
    
    mistakes = response_data.get("grammar_mistakes", [])
    if mistakes:
        console.print("[bold red]Grammar Mistakes:[/bold red]")
        for mistake in mistakes:
            console.print(f"  - {mistake}")
    else:
        console.print("[bold green]No grammar mistakes detected![/bold green]")
        
    console.print(f"\n[bold yellow]Fluency Score:[/bold yellow] {response_data.get('fluency_score', 'N/A')}/10")
    
    console.print(Panel(response_data.get("next_question", "N/A"), title="[bold blue]Next Question[/bold blue]", expand=False))
    console.print("\n")

async def main_loop() -> None:
    """
    Orchestrate the main workflow inside an infinite loop.
    """
    console.print("[bold green]Welcome to Project Echo - AI Interview Assistant[/bold green]")
    
    while True:
        try:
            audio_path = await record_audio()
            response_data = await chat_with_gemini(audio_path)
            render_ui(response_data)
            await play_audio(response_data.get("next_question", ""))
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session ended by user. Goodbye![/bold yellow]")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}")
            console.print(f"[bold red]Error:[/bold red] {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass