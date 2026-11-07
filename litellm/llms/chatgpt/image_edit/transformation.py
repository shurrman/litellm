"""
ChatGPT subscription Images edits API (Codex backend).

ChatGPT accepts JSON body with images:[{image_url: data_url}], not OpenAI
multipart form-data. Convert OpenAI multipart image inputs accordingly.
"""

import base64
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Final, cast

import httpx
from httpx._types import RequestFiles

from litellm.exceptions import AuthenticationError
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class ChatGPTImageEditConfig(BaseImageEditConfig):
    """ChatGPT OAuth-backed image edits for gpt-image-* models."""

    def __init__(self, authenticator: Authenticator | None = None) -> None:
        self.authenticator = authenticator if authenticator is not None else Authenticator()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "image",
            "prompt",
            "background",
            "input_fidelity",
            "mask",
            "model",
            "n",
            "quality",
            "response_format",
            "size",
            "user",
            "extra_headers",
            "timeout",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        return dict(image_edit_optional_params)

    def use_multipart_form_data(self) -> bool:
        return False

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        try:
            access_token: Final = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id: Final = self.authenticator.get_account_id()
        session_id: Final = ensure_chatgpt_session_id(litellm_params or {})
        default_headers: Final = get_chatgpt_default_headers(access_token, account_id, session_id)
        default_headers["accept"] = "application/json"
        default_headers["content-type"] = "application/json"
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        return f"{api_base.rstrip('/')}/images/edits"

    def _read_image_bytes(self, image: object) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, bytearray):
            return bytes(image)
        if isinstance(image, memoryview):
            return image.tobytes()
        if isinstance(image, BytesIO):
            return image.getvalue()
        if isinstance(image, BufferedReader):
            pos = image.tell()
            data = image.read()
            image.seek(pos)
            return data
        if isinstance(image, tuple) and len(image) >= 2:
            return self._read_image_bytes(image[1])
        if hasattr(image, "read"):
            data = image.read()  # type: ignore[attr-defined]
            if isinstance(data, str):
                return data.encode("utf-8")
            return cast(bytes, data)
        if isinstance(image, str):
            if image.startswith("data:"):
                _, _, payload = image.partition(",")
                return base64.b64decode(payload)
            try:
                return base64.b64decode(image, validate=False)
            except Exception:
                return image.encode("utf-8")
        raise TypeError(f"Unsupported image type for ChatGPT edits: {type(image)!r}")

    def _to_data_url(self, image: object) -> str:
        if isinstance(image, str) and image.startswith("data:"):
            return image
        mime = ImageEditRequestUtils.get_image_content_type(image)
        raw = self._read_image_bytes(image)
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        images_payload: list[dict[str, str]] = []
        if image is not None:
            image_list = image if isinstance(image, list) else [image]
            for img in image_list:
                if img is None:
                    continue
                images_payload.append({"image_url": self._to_data_url(img)})

        body: dict[str, object] = {
            "model": model,
            "images": images_payload,
        }
        if prompt is not None:
            body["prompt"] = prompt

        for key in ("n", "size", "quality", "background", "input_fidelity", "response_format", "user"):
            if key in image_edit_optional_request_params and image_edit_optional_request_params[key] is not None:
                body[key] = image_edit_optional_request_params[key]

        if "response_format" not in body:
            body["response_format"] = "b64_json"

        return body, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        try:
            raw_response_json: Final = raw_response.json()
        except Exception:
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            )
        logging_obj.post_call(
            input=model,
            api_key=None,
            original_response=raw_response_json,
        )
        return ImageResponse(**raw_response_json)

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
