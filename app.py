import asyncio
import logging
import os
import json
from typing import Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from rich.console import Console
from rich.panel import Panel
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

# 1. Load environment variables and initialize Gemini client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables.")

async def record_audio() -> str:
    """
    Listen for the Enter key to start and stop audio recording.
    Records audio using sounddevice and saves it to a temporary .wav file.

    Returns:
        str: The file path to the temporary .wav file.
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

async def chat_with_gemini(audio_path: str) -> Dict[str, Any]:
    """
    Upload the .wav file directly to the Gemini Multimodal API.
    Bundles the audio with the Interviewer System Prompt and enforces
    a structured JSON response containing grammar corrections and the next question.

    Args:
        audio_path (str): The path to the temporary audio file.

    Returns:
        Dict[str, Any]: A parsed JSON dictionary containing the evaluation and next question.
    """
    console.print("[cyan]Uploading audio and querying Gemini...[/cyan]")
    
    # Upload file (I/O bound, offloaded to thread)
    audio_file = await asyncio.to_thread(genai.upload_file, audio_path)
    
    prompt = (
        "You are an English technical interviewer. Evaluate the user's spoken response. "
        "Return a JSON object with exactly these keys: "
        "'corrected_sentence' (string, the corrected version of what they said), "
        "'grammar_mistakes' (list of strings, detailing errors), "
        "'fluency_score' (float, 1 to 10), "
        "'next_question' (string, a relevant follow-up technical interview question)."
    )
    
    # Using gemini-1.5-flash as the multimodal API
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        response = await asyncio.to_thread(
            model.generate_content,
            [prompt, audio_file],
            generation_config={"response_mime_type": "application/json"}
        )
        
        result = json.loads(response.text)
        return result
    finally:
        # Clean up the file from Gemini storage
        await asyncio.to_thread(genai.delete_file, audio_file.name)

async def play_audio(text: str) -> None:
    """
    Synthesize and stream-play the provided text using edge-tts.

    Args:
        text (str): The text (e.g., next question) to be converted to speech.
    """
    if not text:
        return

    console.print("[cyan]Synthesizing speech...[/cyan]")
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    audio_file = "response.mp3"
    await communicate.save(audio_file)
    
    console.print("[cyan]Playing audio...[/cyan]")
    # Initialize pygame mixer and play the audio file
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)
        
    pygame.mixer.quit()
    
    # Clean up local audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)

def render_ui(response_data: Dict[str, Any]) -> None:
    """
    Use the rich library to format and print technical grammar corrections,
    fluency metrics, and the next question beautifully to the terminal.

    Args:
        response_data (Dict[str, Any]): The structured response data from Gemini.
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
    Handles the lifecycle: Record -> LLM -> UI -> TTS, with strict exception handling.
    """
    console.print("[bold green]Welcome to Project Echo - AI Interview Assistant[/bold green]")
    
    while True:
        try:
            audio_path = await record_audio()
            response_data = await chat_with_gemini(audio_path)
            render_ui(response_data)
            await play_audio(response_data.get("next_question", ""))
            
            # Clean up temp wav file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session ended by user. Goodbye![/bold yellow]")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}")
            console.print(f"[bold red]Error:[/bold red] {e}")
            await asyncio.sleep(1) # Prevent rapid failure loop

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass
