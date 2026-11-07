"""
Tests for ChatGPT subscription Images generations transformation.
"""

from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.chatgpt.image_generation.transformation import (
    ChatGPTImageGenerationConfig,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


class TestChatGPTImageGenerationConfig:
    def setup_method(self) -> None:
        self.config: Final = ChatGPTImageGenerationConfig()
        self.model: Final = "gpt-image-2"

    def test_provider_config_registration(self) -> None:
        config = ProviderConfigManager.get_provider_image_generation_config(
            model=self.model,
            provider=LlmProviders.CHATGPT,
        )
        assert isinstance(config, ChatGPTImageGenerationConfig)

    def test_supported_params_include_response_format(self) -> None:
        params = self.config.get_supported_openai_params(self.model)
        assert "response_format" in params
        assert "size" in params
        assert "quality" in params

    def test_transform_request_defaults_b64_json(self) -> None:
        body = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a red cube",
            optional_params={"size": "1024x1024"},
            litellm_params={},
            headers={},
        )
        assert body["model"] == "gpt-image-2"
        assert body["prompt"] == "a red cube"
        assert body["response_format"] == "b64_json"
        assert body["size"] == "1024x1024"

    def test_get_complete_url(self) -> None:
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert url.endswith("/images/generations")
        assert "chatgpt.com/backend-api/codex" in url

    def test_validate_environment_sets_json_accept(self) -> None:
        fake_auth = MagicMock()
        fake_auth.get_access_token.return_value = "tok"
        fake_auth.get_account_id.return_value = "acct"
        cfg = ChatGPTImageGenerationConfig(authenticator=fake_auth)
        headers = cfg.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
        )
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["accept"] == "application/json"
        assert headers["content-type"] == "application/json"

    def test_transform_response_parses_b64_data(self) -> None:
        raw = httpx.Response(
            200,
            json={"created": 1, "data": [{"b64_json": "aaa", "revised_prompt": "x"}]},
        )
        out = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=raw,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={"prompt": "a red cube"},
            optional_params={"size": "1024x1024", "quality": "auto"},
            litellm_params={},
            encoding=None,
        )
        assert out.data
        assert out.data[0].b64_json == "aaa"
