"""Pattern review and explanation router."""
from __future__ import annotations

import os
from typing import Any
from fastapi import APIRouter, Form
from core.pattern_discovery import discover_intersectional_patterns
from core.dataset_loader import load_dataset_from_path
from core import store
from core.llm_client import generate

router = APIRouter(prefix="/patterns", tags=["patterns"])

@router.post("/discover")
async def discover_patterns(project_id: int = Form(...)) -> dict[str, Any]:
    project = store.get_project(project_id)
    if not project or not project.get("dataset_path"):
        return {"error": "Project or dataset not found", "patterns": []}

    try:
        df = load_dataset_from_path(project["dataset_path"])
        sensitive_cols = project.get("sensitive_columns", [])
        target_col = project.get("target_column", "")

        if target_col and target_col in df.columns:
            val_counts = df[target_col].value_counts()
            positive_label = val_counts.index[0]
            if 1 in val_counts.index:
                positive_label = 1
            elif 'Yes' in val_counts.index:
                positive_label = 'Yes'

            patterns = discover_intersectional_patterns(df, sensitive_cols, target_col, positive_label)
            return {"patterns": patterns}
        else:
            return {"patterns": []}
    except Exception as e:
        return {"error": str(e), "patterns": []}

@router.post("/explain")
async def explain_pattern(pattern_description: str = Form(...), affected_records: int = Form(...)) -> dict[str, str]:
    prompt = f"""You are an AI fairness expert. A bias detection system found a demographic pattern with high disparity:
Pattern: {pattern_description}
Affected Records: {affected_records}

Explain why dropping these records could be harmful (e.g., loss of representation) or when it might be acceptable. Provide a recommendation on whether to exclude these records from training. Keep it to 2-3 short, plain English paragraphs."""

    try:
        response = generate(prompt, temperature=0.3)
        return {"explanation": response}
    except ImportError:
        return {"explanation": "The openai package is not installed. Run: pip install openai"}
    except Exception as exc:
        return {"explanation": f"Failed to generate explanation: {str(exc)}"}
