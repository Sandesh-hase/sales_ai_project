"""Azure OpenAI client, shared by POST /forecast/explain and POST /chat.

Reads credentials from environment variables (loaded from .env locally via
python-dotenv) -- never hardcode a key here. See .env.example for the
required variable names.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessage

# Load .env from the project root explicitly, rather than relying on the
# process's current working directory (which differs depending on how/where
# the server is launched, e.g. from an IDE run configuration). This file
# lives at backend/src/genai/llm_client.py -- .env is 3 levels up.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

_client = None


class LLMConfigError(RuntimeError):
    """Raised when required Azure OpenAI environment variables are missing."""


def get_client() -> AzureOpenAI:
    """Return a cached AzureOpenAI client, or raise LLMConfigError if unconfigured."""

    global _client

    if _client is None:
        required = {
            "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
            "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise LLMConfigError(f"Missing Azure OpenAI environment variable(s): {', '.join(missing)}")

        _client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    return _client


def call_chat_json(system_prompt: str, user_message: str) -> str:
    """Call the configured Azure OpenAI deployment in JSON mode.

    Returns the raw response content (a JSON string, per response_format).
    Raises LLMConfigError if credentials are missing; propagates the
    underlying openai SDK's exceptions on network/API failures so the caller
    can translate those into a clean HTTP error.
    """

    client = get_client()

    # No `temperature` override -- some deployments (e.g. reasoning-tier models)
    # only support the default value and reject any other.
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


def call_chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    json_mode: bool = False,
) -> ChatCompletionMessage:
    """General-purpose chat completion call, optionally with tool definitions
    and/or JSON-mode output. Used by the /chat tool-calling flow, where the
    caller needs the raw assistant message (to inspect .tool_calls), not
    just its .content -- unlike call_chat_json, which is JSON-mode only and
    used exclusively by /forecast/explain.

    Raises LLMConfigError if credentials are missing; propagates the
    underlying openai SDK's exceptions on network/API failures.
    """

    client = get_client()

    kwargs: dict[str, Any] = {"model": AZURE_OPENAI_DEPLOYMENT_NAME, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick manual test for the Azure OpenAI client.")
    parser.add_argument("--name", default="World", help="Name to greet in the test prompt")
    args = parser.parse_args()

    reply = call_chat_json(
        'You are a helpful assistant. Respond with JSON in this exact shape: {"message": "..."}.',
        f"Say hello to {args.name} in one short sentence.",
    )
    print(reply)
