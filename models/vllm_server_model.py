"""
vLLM server-backed LLM — connects to a running vLLM OpenAI-compatible endpoint
instead of loading the model in-process.

Start the server first:
    bash commands/start_vllm.sh --port 6001 --gpu 0
"""

from openai import OpenAI

from models.base import BaseLLM
from utils import load_config


class VLLMServerModel(BaseLLM):
    """Talks to a vLLM server via the OpenAI-compatible HTTP API."""

    def __init__(self, config: dict | None = None, port: int | None = None):
        if config is None:
            config = load_config()

        model_cfg = config.get("model", {})
        server_cfg = config.get("vllm_server", {})

        self.model_path: str = model_cfg["local_model_path"]
        _port = port if port is not None else server_cfg.get("port", 6001)
        host = server_cfg.get("host", "127.0.0.1")

        self._temperature: float = model_cfg.get("temperature", 0.7)
        self._top_p: float = model_cfg.get("top_p", 0.8)

        self._client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{host}:{_port}/v1",
            timeout=600.0,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> tuple[list, str, str]:
        """Send a chat completion request to the vLLM server.

        Returns:
            ([], raw_text, clean_text)  — token_ids not available from HTTP API
        """
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=max_new_tokens,
        )
        text = response.choices[0].message.content or ""
        return [], text, text

    def clear_cache(self) -> None:
        pass  # server manages its own KV cache
