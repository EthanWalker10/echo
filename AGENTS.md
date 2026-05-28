# Project Echo

AI-native voice-based English technical interview assistant.

## Engineering Philosophy
- MVP first: Pure Terminal/CLI interface.
- Avoid over-engineering: No web frameworks, no databases.
- Keep latency low: Optimize API calls and audio streams.
- Simplicity over abstraction.

## Stack
- Python 3.11
- Package Manager: Poetry
- Gemini gemini-3.5-flash API (Multimodal Audio Input & LLM Core)
- Edge-TTS (TTS)
- Local JSON storage

## Rules
- Use async for all I/O operations.
- Keep functions small and composable.
- Every external API call must have robust timeout and error handling.
- Use structured logging.

## AI Coding Behavior
- Do not refactor unrelated modules.
- Keep code changes minimal.
- Preserve the current single-monolith, CLI-focused project structure.