
"""Azure AI Foundry OpenAI client.

- POST /forecast/explain: Responses API
- POST /chat: Chat Completions API

Credentials are loaded from the backend .env file.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


# Load .env from the backend project root.
load_dotenv(
    Path(__file__).resolve().parent.parent.parent / ".env"
)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_NAME"
)

_client: Optional[OpenAI] = None


class LLMConfigError(RuntimeError):
    """Raised when required Azure AI Foundry settings are missing."""


def get_client() -> OpenAI:
    """Return a cached Azure AI Foundry OpenAI client."""

    global _client

    if _client is None:
        required = {
            "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
            "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
        }

        missing = [
            name for name, value in required.items() if not value
        ]

        if missing:
            raise LLMConfigError(
                f"Missing environment variable(s): {', '.join(missing)}"
            )

        endpoint = AZURE_OPENAI_ENDPOINT.strip().rstrip("/")

        if not endpoint.endswith("/openai/v1"):
            endpoint += "/openai/v1"

        _client = OpenAI(
            base_url=endpoint,
            api_key=AZURE_OPENAI_API_KEY,
            timeout=60.0,
            max_retries=2,
        )

    return _client


def call_chat_json(system_prompt: str, user_message: str) -> str:
    """Generate a JSON response using the Responses API.

    Used by POST /forecast/explain.
    """

    client = get_client()

    response = client.responses.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        input=[
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\nReturn a valid JSON object without Markdown."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        text={
            "format": {
                "type": "json_object"
            }
        },
    )

    content = response.output_text

    if not content:
        raise RuntimeError(
            "Azure AI Foundry returned an empty response."
        )

    # Validate that the model returned valid JSON.
    json.loads(content)

    return content


def call_chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    json_mode: bool = False,
) -> ChatCompletionMessage:
    """Generate a Chat Completions response.

    Used by POST /chat and its existing tool-calling flow.
    """

    client = get_client()

    kwargs: dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT_NAME,
        "messages": messages,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    if json_mode:
        kwargs["response_format"] = {
            "type": "json_object"
        }

    response = client.chat.completions.create(**kwargs)

    return response.choices[0].message


if __name__ == "__main__":
    result = call_chat_json(
        system_prompt=(
            "You are a helpful assistant. "
            'Respond in JSON with this structure: {"message": "..."}'
        ),
        user_message="What is the capital of France?",
    )

    print(json.dumps(json.loads(result), indent=2))
