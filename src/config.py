"""
config.py — Configuration loader.

Loads and validates the YAML config file, providing defaults
for any missing values.
"""

from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration from YAML file with sensible defaults."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Apply defaults for any missing keys
    defaults = {
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
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        "evaluation": {
            "golden_qa_path": "evals/golden_qa.json",
            "faithfulness_threshold": 0.85,
            "citation_coverage_threshold": 0.90,
        },
        "observability": {"log_dir": "./logs", "log_requests": True, "log_level": "INFO"},
        "app": {"host": "0.0.0.0", "port": 8000, "title": "DocuQuery"},
    }

    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = section_defaults
        elif isinstance(section_defaults, dict):
            for key, value in section_defaults.items():
                config[section].setdefault(key, value)

    return config
