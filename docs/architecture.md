# Architecture Overview

## Project Name

Project Echo

---

# Product Goal

A low-latency AI-powered English interview practice assistant.

The product focuses on:

- real-time speech interaction
- grammar correction
- interview simulation
- conversational memory
- ultra-simple MVP architecture

Target users:

- software engineers
- AI engineers
- international interview candidates

---

# Core Principles

## MVP First

This project is MVP-oriented.

Avoid:

- microservices
- distributed systems
- over-abstraction
- premature optimization
- web frameworks (e.g., FastAPI) at this stage

Prefer:

- simple modules
- readable code
- direct implementation
- fast iteration
- pure terminal (CLI) execution

---

## AI-Native Development

This project is designed for AI-assisted coding.

Requirements:

- clear module boundaries
- deterministic naming
- minimal hidden magic
- explicit data flow
- highly readable structure

All architecture decisions should optimize for:

1. AI context understanding
2. code generation consistency
3. maintainability
4. debugging simplicity

---

# Tech Stack

## Backend

- Python 3.11
- Package Manager: Poetry

## AI Services

- Gemini 1.5 Pro/Flash API (Multimodal: handles both Audio Input and Text Generation)
- Edge-TTS (for Text-to-Speech)

## Storage

- local JSON storage
- optional SQLite later

## Frontend

Current stage:

- terminal interface (CLI)

Future stage:

- Next.js web app

---

# System Architecture

## Modules

### 1. Audio Input Layer

Responsibilities:

- microphone recording
- audio preprocessing
- temporary .wav file handling

Main files:

- src/audio/recorder.py

---

### 2. Multimodal Conversation Engine (Gemini)

Responsibilities:

- manage dialogue history
- construct prompts
- upload temporary .wav file
- call Gemini Multimodal API (Audio + Text Prompt)
- parse structured JSON responses

Main files:

- src/llm/chat_engine.py

---

### 3. Text-to-Speech Layer

Responsibilities:

- generate speech audio from LLM's text response
- playback synthesized voice

Main files:

- src/tts/edge_tts.py

---

### 4. Session Storage

Responsibilities:

- save dialogue history
- persist interview sessions

Main files:

- data/session_log.json

---

# Data Flow

```text
Microphone
    ↓
Audio Recorder (saves as temp.wav)
    ↓
gemini-3.1-flash-lite Multimodal API (Audio + System Prompt)
    ↓
Structured JSON Response (Correction & Next Question)
    ↓
Terminal Rendering (Rich UI)
    ↓
Edge-TTS
    ↓
Audio Playback



# API Response Contract

LLM responses MUST return JSON.

Example:
```json
{
  "corrected_sentence": "...",
  "grammar_mistakes": [],
  "fluency_score": 8.5,
  "next_question": "..."
}
```

Never return free-form text.

# Coding Standards
## General Rules
- Keep functions small
- Prefer explicit logic
- Avoid hidden side effects
- Use type hints everywhere
- Use dataclasses or Pydantic models
## Error Handling

All IO operations MUST include:

- try/except
- logging
- meaningful error messages

Never silently ignore exceptions.

## Logging

Use standard logging module.

Forbidden:
- print debugging
- random console output

## File Organization

One responsibility per module.

Avoid giant files.

Recommended:
- <300 lines per file

# AI Coding Rules
When generating code

The AI assistant MUST:
- preserve module boundaries
- avoid unnecessary abstractions
- avoid introducing frameworks
- avoid speculative architecture
When refactoring

The AI assistant MUST:
- preserve behavior
- minimize changes
- explain architectural impact

# Future Expansion
Potential future features:
- WebSocket streaming
- real-time voice interruption
- emotion detection
- personalized interview profiles
- vector memory
- multi-agent coaching
- mobile app

Current MVP should NOT implement these features yet.

# Non-Goals
Current MVP does NOT require:
- Kubernetes
- Redis cluster
- distributed queue
- multi-tenant architecture
- RBAC
- CI/CD complexity

Keep the system lightweight.