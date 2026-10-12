"""
test_generator.py — Provider abstraction, prompt building and answer metadata.
Run: pytest tests/test_generator.py -v
"""

import pytest

from src.config import load_config
from src.generator import AnthropicClient, Generator, GroqClient, make_llm_client
from tests.fakes import FakeLLM, make_chunk, sdk_ns

CONFIG = {"generation": {"provider": "fake", "model": "m", "temperature": 0.1,
                         "max_tokens": 100, "system_prompt": "SYS"}}


class TestGenerator:
    def test_answer_metadata_and_invalid_citations(self):
        chunks = [make_chunk(cid="abc12345" + "0" * 8), make_chunk(cid="def67890" + "0" * 8)]
        llm = FakeLLM(["Claim one is grounded here [abc12345]. Claim two is made up here [ffffffff]."])
        out = Generator(CONFIG, llm=llm).generate("question?", chunks)

        assert out.citations == ["abc12345", "ffffffff"]
        assert out.invalid_citations == ["ffffffff"]
        assert out.citation_coverage == 1.0
        assert out.total_tokens == 60
        assert out.model == "fake/m"

    def test_prompt_contains_labelled_chunks_and_system_prompt(self):
        llm = FakeLLM(["ok"])
        Generator(CONFIG, llm=llm).generate("What is X?", [make_chunk(cid="abc12345" + "0" * 8, text="X is Y.")])
        call = llm.calls[0]
        assert call["system"] == "SYS"
        assert "Chunk [abc12345]" in call["user"] and "X is Y." in call["user"]
        assert "Question: What is X?" in call["user"]


class TestProviderClients:
    def test_groq_wrapper(self):
        sdk = sdk_ns(chat=sdk_ns(completions=sdk_ns(create=lambda **kw: sdk_ns(
            choices=[sdk_ns(message=sdk_ns(content="hi"))],
            usage=sdk_ns(prompt_tokens=3, completion_tokens=4)))))
        out = GroqClient(client=sdk).complete("s", "u", "m", 0.1, 10)
        assert (out.text, out.prompt_tokens, out.completion_tokens) == ("hi", 3, 4)

    def test_anthropic_wrapper(self):
        seen = {}

        def create(**kw):
            seen.update(kw)
            return sdk_ns(content=[sdk_ns(type="text", text="hello")],
                          usage=sdk_ns(input_tokens=7, output_tokens=2))

        out = AnthropicClient(client=sdk_ns(messages=sdk_ns(create=create))).complete("s", "u", "m", 0.1, 10)
        assert (out.text, out.prompt_tokens, out.completion_tokens) == ("hello", 7, 2)
        assert seen["system"] == "s" and seen["messages"] == [{"role": "user", "content": "u"}]

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            make_llm_client("nope")

    def test_missing_key_is_clear(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            make_llm_client("anthropic")


class TestConfig:
    def test_defaults_present(self):
        cfg = load_config()
        assert cfg["generation"]["provider"] == "groq"
        assert cfg["evaluation"]["answer_correctness_threshold"] == 0.75

    def test_env_switches_provider_and_default_model(self, monkeypatch):
        monkeypatch.setenv("DOCUQUERY_LLM_PROVIDER", "anthropic")
        cfg = load_config()
        assert cfg["generation"]["provider"] == "anthropic"
        assert cfg["generation"]["model"].startswith("claude-")

    def test_env_model_and_paths(self, monkeypatch):
        monkeypatch.setenv("DOCUQUERY_LLM_MODEL", "custom-model")
        monkeypatch.setenv("DOCUQUERY_CHROMA_DIR", "/data/chroma")
        cfg = load_config()
        assert cfg["generation"]["model"] == "custom-model"
        assert cfg["vector_store"]["persist_directory"] == "/data/chroma"

    def test_bad_provider_rejected(self, monkeypatch):
        monkeypatch.setenv("DOCUQUERY_LLM_PROVIDER", "openai")
        with pytest.raises(ValueError):
            load_config()
