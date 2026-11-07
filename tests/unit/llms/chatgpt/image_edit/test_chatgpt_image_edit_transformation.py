"""
Tests for ChatGPT subscription Images edits transformation.
"""

import base64
from typing import Final
from unittest.mock import MagicMock

import httpx

from litellm.llms.chatgpt.image_edit.transformation import ChatGPTImageEditConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


class TestChatGPTImageEditConfig:
    def setup_method(self) -> None:
        self.config: Final = ChatGPTImageEditConfig()
        self.model: Final = "gpt-image-2"

    def test_provider_config_registration(self) -> None:
        config = ProviderConfigManager.get_provider_image_edit_config(
            model=self.model,
            provider=LlmProviders.CHATGPT,
        )
        assert isinstance(config, ChatGPTImageEditConfig)

    def test_uses_json_not_multipart(self) -> None:
        assert self.config.use_multipart_form_data() is False

    def test_transform_request_converts_bytes_to_data_url(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"0" * 16
        body, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt="make it blue",
            image=[png],
            image_edit_optional_request_params={"size": "1024x1024"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert files == []
        assert body["model"] == "gpt-image-2"
        assert body["prompt"] == "make it blue"
        assert body["response_format"] == "b64_json"
        assert body["images"][0]["image_url"].startswith("data:image/")
        encoded = body["images"][0]["image_url"].split(",", 1)[1]
        assert base64.b64decode(encoded) == png

    def test_get_complete_url(self) -> None:
        url = self.config.get_complete_url(
            model=self.model,
            api_base=None,
            litellm_params={},
        )
        assert url.endswith("/images/edits")

    def test_transform_response(self) -> None:
        raw = httpx.Response(
            200,
            json={"created": 1, "data": [{"b64_json": "bbb"}]},
        )
        out = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=raw,
            logging_obj=MagicMock(),
        )
        assert isinstance(out, ImageResponse)
        assert out.data
        assert out.data[0].b64_json == "bbb"
