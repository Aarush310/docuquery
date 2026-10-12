"""
config.py — Configuration loader.

Loads the YAML config, fills in defaults for anything missing, then applies
environment-variable overrides so the same image can be configured per
environment (local, CI, container, OpenShift) without editing files.
"""

import copy
import os
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

# Default model for each supported LLM provider.
PROVIDER_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-haiku-5-5",
}

DEFAULTS = {
    "chunking": {"chunk_size": 600, "chunk_overlap": 100, "min_chunk_size": 50},
    "embedding": {"model": "all-MiniLM-L6-v2"},
    "vector_store": {"collection_name": "docuquery", "persist_directory": "./chroma_db"},
    "retrieval": {
        "top_k": 5,
        "bm25_weight": 0.3,
        "vector_weight": 0.7,
        "reranker": {
            "enabled": True,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_n": 3,
        },
    },
    "generation": {
        "provider": "groq",
        "model": PROVIDER_DEFAULT_MODELS["groq"],
        "temperature": 0.1,
        "max_tokens": 1024,
    },
    "evaluation": {
        "golden_qa_path": "evals/golden_qa.json",
        "results_path": "evals/results.json",
        "citation_coverage_threshold": 0.90,
        "faithfulness_threshold": 0.85,
        "answer_correctness_threshold": 0.75,
        "latency_p95_ms_threshold": 8000,
        "max_error_rate": 0.0,
    },
    "observability": {
        "log_dir": "./logs",
        "log_requests": True,
        "log_level": "INFO",
        "max_traces_in_memory": 1000,
        "alerts": {"min_citation_coverage": 0.90, "max_latency_ms": 8000},
    },
    "ingest": {"allowed_root": "data"},
    "app": {"host": "0.0.0.0", "port": 8000, "title": "DocuQuery"},
}

# Environment variable -> (section, key)
ENV_OVERRIDES = {
    "DOCUQUERY_CHROMA_DIR": ("vector_store", "persist_directory"),
    "DOCUQUERY_LOG_DIR": ("observability", "log_dir"),
    "DOCUQUERY_LOG_LEVEL": ("observability", "log_level"),
    "DOCUQUERY_DATA_DIR": ("ingest", "allowed_root"),
    "DOCUQUERY_LLM_MODEL": ("generation", "model"),
}


def _merge_defaults(config: dict, defaults: dict) -> dict:
    for key, value in defaults.items():
        if key not in config or config[key] is None:
            config[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(config[key], dict):
            _merge_defaults(config[key], value)
    return config


def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration from YAML file with defaults and env overrides."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    _merge_defaults(config, DEFAULTS)

    # Switching provider via env also switches to that provider's default model,
    # unless DOCUQUERY_LLM_MODEL pins a specific one.
    provider = os.environ.get("DOCUQUERY_LLM_PROVIDER")
    if provider:
        provider = provider.lower()
        if provider not in PROVIDER_DEFAULT_MODELS:
            raise ValueError(
                f"Unknown DOCUQUERY_LLM_PROVIDER={provider!r}; "
                f"expected one of {sorted(PROVIDER_DEFAULT_MODELS)}"
            )
        if provider != config["generation"].get("provider"):
            config["generation"]["model"] = PROVIDER_DEFAULT_MODELS[provider]
        config["generation"]["provider"] = provider

    for env_var, (section, key) in ENV_OVERRIDES.items():
        if os.environ.get(env_var):
            config[section][key] = os.environ[env_var]

    return config
