"""Chatbot router for context-aware assistance."""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from core.llm_client import generate

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: dict[str, Any] = {}

def _build_chat_prompt(req: ChatRequest) -> str:
    """Build a prompt incorporating chat history and current context."""
    context_json = json.dumps(req.context, indent=2, default=str) if req.context else "None"

    prompt = f"""You are a helpful and knowledgeable AI Fairness Assistant for the 'Bias-Lab' platform.
Your job is to help the user understand their bias metrics, fairness scores, and mitigation options.

CURRENT USER CONTEXT:
{context_json}

INSTRUCTIONS:
- If the user asks about a specific metric or chart, refer to the CURRENT USER CONTEXT.
- Keep your answers concise, practical, and easy to understand for non-technical users.
- Do NOT hallucinate metrics that are not present in the context.
- Use plain English and short paragraphs.

CONVERSATION HISTORY:
"""
    for msg in req.messages:
        role = "User" if msg.role == "user" else "Assistant"
        prompt += f"\n{role}: {msg.content}"

    prompt += "\n\nAssistant:"
    return prompt

@router.post("")
async def chat_with_assistant(req: ChatRequest) -> dict[str, str]:
    try:
        prompt = _build_chat_prompt(req)
        response = generate(prompt, temperature=0.3)
        return {"response": response, "status": "ok"}
    except ImportError:
        return {"response": "Error: openai package is not installed. Run: pip install openai", "status": "import_error"}
    except Exception as exc:
        return {"response": f"Error: {str(exc)}", "status": "error"}
