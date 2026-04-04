"""
vLLM-backed LLM implementation.
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .base import BaseLLM
from ..utils import load_config


class VLLMModel(BaseLLM):
    """Wraps vLLM for local model inference."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        cfg = config["model"]

        self.model_path: str = cfg["local_model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
            max_model_len=cfg.get("max_model_len", 8192),
        )
        self._temperature: float = cfg.get("temperature", 0.7)
        self._top_p: float = cfg.get("top_p", 0.8)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> tuple[list, str, str]:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        params = SamplingParams(
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        outputs = self.llm.generate([formatted], params)
        output = outputs[0]
        clean_text = output.outputs[0].text
        token_ids = list(output.outputs[0].token_ids)
        return token_ids, clean_text, clean_text

    def clear_cache(self) -> None:
        pass  # vLLM manages KV cache automatically
