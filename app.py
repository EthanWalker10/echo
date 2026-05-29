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

# Memory Architecture Paths
MEMORY_DIR = "memory"
SESSION_FILE = os.path.join(MEMORY_DIR, "current_session.json")
COMPRESSED_FILE = os.path.join(MEMORY_DIR, "compressed_memory.json")
METRICS_FILE = os.path.join(MEMORY_DIR, "metrics.json")

def init_memory_dir():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)

init_memory_dir()

async def read_json(filepath: str, default: Any) -> Any:
    def _read():
        if not os.path.exists(filepath):
            return default
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default
    return await asyncio.to_thread(_read)

async def write_json(filepath: str, data: Any):
    def _write():
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return await asyncio.to_thread(_write)

# 1. Load environment variables and initialize Gemini Client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
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

class VocabMistake(BaseModel):
    wrong_phrase: str
    correct_phrase: str

class InterviewResponse(BaseModel):
    corrected_sentence: str
    grammar_mistakes: List[str]
    fluency_score: float
    tech_correctness_score: float
    tech_feedback: str
    identified_weak_grammar: List[str] = Field(default_factory=list)
    identified_weak_topics: List[str] = Field(default_factory=list)
    identified_weak_tech_concepts: List[str] = Field(default_factory=list)
    identified_topics_discussed: List[str] = Field(default_factory=list)
    identified_vocab_mistakes: List[VocabMistake] = Field(default_factory=list)
    resolved_weak_tech_concepts: List[str] = Field(default_factory=list)
    resolved_weak_grammar: List[str] = Field(default_factory=list)
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
    
    audio_file = await asyncio.to_thread(client.files.upload, file=audio_path)
    
    session_data = await read_json(SESSION_FILE, [])
    compressed_data = await read_json(COMPRESSED_FILE, {})
    
    prompt = (
        "You are an English technical interviewer. Evaluate the user's spoken response.\n\n"
        "### User's Memory Profile (Weaknesses & Topics to Focus On):\n"
        f"{json.dumps(compressed_data, indent=2)}\n\n"
        "### Recent Dialogue History (Last 3 Rounds):\n"
        f"{json.dumps(session_data, indent=2)}\n\n"
        "### Instructions:\n"
        "CRITICAL: You must ONLY evaluate what the user actually said in the audio regarding the CURRENT question. "
        "Do NOT penalize the user for concepts or standards that you plan to ask in 'next_question' but haven't been tested yet.\n"
        "If the user scores high (e.g., tech_correctness_score >= 8.5), aggressively check if they have now mastered any of their past 'weak_tech_concepts' or 'weak_grammar' and list them in resolved_weak_tech_concepts or resolved_weak_grammar.\n"
        "Evaluate the current audio input based on the history and profile. "
        "Return a JSON object exactly matching the schema. "
        "Extract newly identified weak grammar, weak topics, weak tech concepts, topics discussed, "
        "and vocab mistakes (as a list of objects with wrong_phrase and correct_phrase)."
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
    except Exception as e:
        logger.error(f"Gemini API generation failed: {e}")
        raise
    finally:
        await asyncio.to_thread(client.files.delete, name=audio_file.name)

async def update_memory_background(response_data: dict):
    """
    Background hook to parse evaluation, update metrics, and compress memory
    using non-blocking async file writes.
    """
    try:
        # 1. Update metrics
        metrics = await read_json(METRICS_FILE, {"fluency_average": 0.0, "total_rounds": 0})
        total = metrics.get("total_rounds", 0)
        avg = metrics.get("fluency_average", 0.0)
        new_score = response_data.get("fluency_score", 0.0)
        
        metrics["fluency_average"] = (avg * total + new_score) / (total + 1)
        metrics["total_rounds"] = total + 1
        await write_json(METRICS_FILE, metrics)

        # 2. Update compressed memory
        compressed = await read_json(COMPRESSED_FILE, {
            "weak_grammar": [],
            "weak_topics": [],
            "weak_tech_concepts": [],
            "topics_discussed": [],
            "common_vocab_mistakes": {},
            "fluency_average": 0.0,
            "mastered_grammar": [],
            "mastered_tech_concepts": []
        })
        
        # Ensure new fields exist in case of old JSON file
        if "mastered_grammar" not in compressed:
            compressed["mastered_grammar"] = []
        if "mastered_tech_concepts" not in compressed:
            compressed["mastered_tech_concepts"] = []
            
        def merge_lists(key: str, new_items: list):
            if new_items:
                compressed[key].extend(new_items)
                compressed[key] = list(set(compressed[key]))

        def resolve_weaknesses(weak_key: str, mastered_key: str, resolved_items: list):
            if resolved_items:
                compressed[mastered_key].extend(resolved_items)
                compressed[mastered_key] = list(set(compressed[mastered_key]))
                compressed[weak_key] = [item for item in compressed[weak_key] if item not in resolved_items]
                
        merge_lists("weak_grammar", response_data.get("identified_weak_grammar", []))
        merge_lists("weak_topics", response_data.get("identified_weak_topics", []))
        merge_lists("weak_tech_concepts", response_data.get("identified_weak_tech_concepts", []))
        merge_lists("topics_discussed", response_data.get("identified_topics_discussed", []))
        
        resolve_weaknesses("weak_grammar", "mastered_grammar", response_data.get("resolved_weak_grammar", []))
        resolve_weaknesses("weak_tech_concepts", "mastered_tech_concepts", response_data.get("resolved_weak_tech_concepts", []))
        
        vocab_mistakes = response_data.get("identified_vocab_mistakes", [])
        for mistake in vocab_mistakes:
            wrong = mistake.get("wrong_phrase")
            correct = mistake.get("correct_phrase")
            if wrong and correct:
                compressed["common_vocab_mistakes"][wrong] = correct
                
        compressed["fluency_average"] = round(metrics["fluency_average"], 2)
            
        await write_json(COMPRESSED_FILE, compressed)

        # 3. Update session (sliding window of 3 rounds)
        session = await read_json(SESSION_FILE, [])
        session.append({
            "user_said": response_data.get("corrected_sentence", ""),
            "assistant_asked": response_data.get("next_question", "")
        })
        if len(session) > 3:
            session = session[-3:]
        await write_json(SESSION_FILE, session)
        
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")

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
    
    tech_score = response_data.get("tech_correctness_score", "N/A")
    tech_feedback = response_data.get("tech_feedback", "N/A")
    console.print(f"\n[bold magenta]Tech Correctness Score:[/bold magenta] {tech_score}/10")
    console.print(Panel(tech_feedback, title="[bold magenta]Tech Feedback[/bold magenta]", expand=False))
    
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
            
            # Fire and forget background memory update
            asyncio.create_task(update_memory_background(response_data))
            
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
