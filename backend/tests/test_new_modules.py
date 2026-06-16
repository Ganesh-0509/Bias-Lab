"""Tests for newly added modules: dataset_loader, colab, llm_narrative."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.dataset_loader import list_datasets, load_dataset


class TestDatasetLoader:
    def test_list_datasets_returns_expected(self):
        datasets = list_datasets()
        names = [d["name"] for d in datasets]
        assert "adult_income" in names
        assert "compas" in names

    def test_list_datasets_has_required_fields(self):
        for ds in list_datasets():
            assert "name" in ds
            assert "display_name" in ds
            assert "target_col" in ds
            assert "sensitive_cols" in ds
            assert "rows" in ds
            assert isinstance(ds["available"], bool)

    def test_load_adult_income(self):
        df = load_dataset("adult_income")
        assert df.shape[0] > 48000
        assert "income" in df.columns
        assert "race" in df.columns
        assert "sex" in df.columns

    def test_load_compas(self):
        df = load_dataset("compas")
        assert df.shape[0] > 7000
        assert "two_year_recid" in df.columns
        assert "race" in df.columns

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent")


class TestColabExport:
    def test_export_returns_valid_notebook(self):
        from routers.colab import export_notebook

        payload = {
            "results": {
                "fairness_score": 72,
                "decision": "MODERATE RISK",
                "scores": {
                    "data_bias_score": 80,
                    "model_bias_score": 65,
                    "proxy_risk_score": 70,
                },
                "recommendations": [
                    {"fix_id": "fix_1", "title": "Fix 1", "description": "Test fix"}
                ],
            },
            "dataset_name": "test_data",
        }

        import asyncio
        response = asyncio.run(export_notebook(payload))
        assert response.media_type == "application/x-ipynb+json"
        notebook = json.loads(response.body)
        assert notebook["nbformat"] == 4
        assert len(notebook["cells"]) > 3


class TestLLMNarrative:
    def test_build_prompt_includes_scores(self):
        from routers.llm_narrative import _build_prompt

        results = {
            "fairness_score": 72,
            "decision": "MODERATE RISK",
            "scores": {
                "data_bias_score": 80,
                "model_bias_score": 65,
                "proxy_risk_score": 70,
                "counterfactual_score": 75,
                "stress_test_score": 70,
            },
            "recommendations": [
                {"fix_id": "fix_1", "title": "Threshold Tuning", "description": "Adjust decision threshold"}
            ],
            "explain_summary": "Key factors: credit score, income.",
        }

        prompt = _build_prompt(results)
        assert "Fairness Score: 72" in prompt
        assert "MODERATE RISK" in prompt
        assert "Threshold Tuning" in prompt
        assert "Data Bias" in prompt

    def test_build_prompt_empty_recommendations(self):
        from routers.llm_narrative import _build_prompt

        results = {
            "fairness_score": 95,
            "decision": "LOW RISK",
            "scores": {"data_bias_score": 95, "model_bias_score": 95},
            "recommendations": [],
            "explain_summary": "",
        }

        prompt = _build_prompt(results)
        assert "No specific recommendations" in prompt

    def test_endpoint_no_key_returns_friendly_message(self):
        from routers.llm_narrative import generate_narrative
        import asyncio

        resp = asyncio.run(generate_narrative({"results": {}}))
        assert resp["status"] in ("import_error", "ollama_unreachable", "error")
