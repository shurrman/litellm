"""
ChatGPT subscription Images API (Codex backend).

Proxies OpenAI-compatible /images/generations to
https://chatgpt.com/backend-api/codex/images/generations using the same
OAuth session as chatgpt responses models.
"""

from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer
else:
    LiteLLMLoggingObj = Any


class ChatGPTImageGenerationConfig(BaseImageGenerationConfig):
    """ChatGPT OAuth-backed image generation for gpt-image-* models."""

    def __init__(self, authenticator: Authenticator | None = None) -> None:
        self.authenticator = authenticator if authenticator is not None else Authenticator()

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return [
            "background",
            "moderation",
            "n",
            "output_compression",
            "output_format",
            "quality",
            "response_format",
            "size",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)
        for k in non_default_params:
            if k not in optional_params:
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. "
                        f"Supported parameters are {supported_params}. "
                        "Set drop_params=True to drop unsupported parameters."
                    )
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
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
        session_id: Final = ensure_chatgpt_session_id(litellm_params)
        default_headers: Final = get_chatgpt_default_headers(access_token, account_id, session_id)
        # Images API returns JSON, not SSE.
        default_headers["accept"] = "application/json"
        default_headers["content-type"] = "application/json"
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        return f"{api_base.rstrip('/')}/images/generations"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        body: Final = {"model": model, "prompt": prompt, **optional_params}
        if "response_format" not in body:
            body["response_format"] = "b64_json"
        return body

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response: Final = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=(
                    f"Error parsing ChatGPT image generation response: {e}; "
                    f"body={raw_response.text[:500]}"
                ),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response,
        )

        if isinstance(response, dict) and isinstance(response.get("data"), list):
            image_response: Final[ImageResponse] = convert_to_model_response_object(
                response_object=response,
                model_response_object=model_response,
                response_type="image_generation",
            )
        else:
            if not model_response.data:
                model_response.data = []
            image_response = model_response

        image_response.size = image_response.size or optional_params.get("size", "1024x1024")
        image_response.quality = image_response.quality or optional_params.get("quality", "auto")
        image_response.output_format = image_response.output_format or optional_params.get(
            "output_format", "png"
        )
        return image_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
