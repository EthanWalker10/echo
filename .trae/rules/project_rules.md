# Project Rules

## Architecture
- This project is a strict MVP.
- Build a pure Terminal (CLI) application. Do NOT introduce web frameworks (like FastAPI) or microservices.
- Prioritize ultra-low latency and highly readable code.

## Backend & Tech Stack
- Language: Python 3.11 (managed by `Poetry`).
- Core Pipeline: Audio File (.wav) -> gemini-3.1-flash-lite Multimodal API (Audio Input) -> Edge-TTS (TTS).
- Async I/O: All network calls and audio processing must use async/await with strict timeout handling.

## Memory
- Use lightweight local JSON files for session persistence.
- Do NOT introduce SQLite or any other relational databases at this stage.

## Logging & Output
- Use Python's standard `logging` module for structured logs.
- Do not use `print()` for debugging. Use `print()` (or UI libraries like `rich`) ONLY for user-facing terminal interaction.