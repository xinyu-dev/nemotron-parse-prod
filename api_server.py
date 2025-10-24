# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This file is derived from: https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py
#
#   Copyright 2024,2024 vLLM Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import gc

import nim_llm_sdk.envs as envs
import numpy as np

if envs.NIM_TRUST_CUSTOM_CODE or envs.NIM_ENABLE_BUDGET_CONTROL:
    from nim_llm_sdk.custom_guided_decoding.patch import CustomGuidedDecodingLoader

    CustomGuidedDecodingLoader.maybe_patch()

import asyncio
import json
import os
import signal
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from http import HTTPStatus
from textwrap import dedent
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from unittest.mock import MagicMock

import fastapi
import nim_llm_sdk
import nvtx
import torch

# CRITICAL: Early torchao.quantization import prevents "Cannot copy out of meta tensor" errors with quantized models in vLLM
import torchao.quantization  # noqa: F401
import uvicorn
import uvloop
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from mpi4py import MPI
from nim_llm_sdk.custom_guided_decoding import maybe_set_custom_parameters, maybe_update_constrained_response
from nim_llm_sdk.engine.arg_utils import NimAsyncEngineArgs
from nim_llm_sdk.engine.async_trtllm_engine_factory import AsyncLLMEngineFactory
from nim_llm_sdk.engine.trtllm_errors import LoraCacheFullError
from nim_llm_sdk.entrypoints.args import extract_lora_config, prepare_environment
from nim_llm_sdk.entrypoints.llamastack.protocol import ChatCompletionRequest as LlamaStackChatCompletionRequest
from nim_llm_sdk.entrypoints.llamastack.protocol import ChatCompletionResponse as LlamaStackChatCompletionResponse
from nim_llm_sdk.entrypoints.llamastack.protocol import (
    ChatCompletionResponseStreamChunk as LlamaStackChatCompletionResponseStreamChunk,
)
from nim_llm_sdk.entrypoints.llamastack.protocol import CompletionRequest as LlamaStackCompletionRequest
from nim_llm_sdk.entrypoints.llamastack.protocol import CompletionResponse as LlamaStackCompletionResponse
from nim_llm_sdk.entrypoints.llamastack.protocol import (
    CompletionResponseStreamChunk as LlamaStackCompletionResponseStreamChunk,
)
from nim_llm_sdk.entrypoints.llamastack.serving_chat import LlamaStackServingChat
from nim_llm_sdk.entrypoints.llamastack.serving_completion import LlamaStackServingCompletion
from nim_llm_sdk.entrypoints.openai.api_extensions import (
    NIMHealthSuccessResponse,
    NIMLicenseInfoResponse,
    NIMLLMChatCompletionRequest,
    NIMLLMCompletionRequest,
    NIMLLMVersionResponse,
    NIMMetadataResponse,
    NIMModelInfoResponse,
    OpenAIServingChat,
    OpenAIServingCompletion,
)
from nim_llm_sdk.entrypoints.openai.middleware.field_transformation import FieldTransformationMiddleware
from nim_llm_sdk.entrypoints.openai.middleware.prompt_telemetry import PromptTelemetryMiddleware
from nim_llm_sdk.entrypoints.openai.prompt_telemetry import is_prompt_telemetry_enabled, set_prompt_telemetry_webhooks
from nim_llm_sdk.hub.local_cache_manager import cache_model
from nim_llm_sdk.hub.ngc_injector import inject_ngc_hub
from nim_llm_sdk.hub.profile_utils import is_trt_llm_model
from nim_llm_sdk.logger import (
    configure_all_loggers_with_handlers_except,
    configure_logger,
    get_logging_config_for_package,
    init_logger,
)
from nim_llm_sdk.model_specific_modules.dynamic_module_loader import load_dynamic_modules
from nim_llm_sdk.patch.openai_api_server import apply_custom_model_parameters
from nim_llm_sdk.sampling_params import SamplingParamsExt as SamplingParams
from nim_llm_sdk.trtllm.llm_api_utils import LLMAPIValueError

# import this first due to issues with mpi4py
from nimlib.nim_inference_api_builder.http_api import HttpNIMApiInterface
from nvext_peft.model_synchronizers import init_model_synchronizers
from PIL import Image
from prometheus_client import REGISTRY
from starlette.responses import Response as StarletteResponse
from typing_extensions import override
from vllm.entrypoints.chat_utils import MultiModalItemTracker, load_chat_template
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionResponse,
    CompletionStreamResponse,
    ErrorResponse,
    ModelList,
)
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import find_process_using_port

load_dynamic_modules()


# Image format validation functions fd 
def validate_image_url_format(url: str) -> None:
    """Check if URL has a supported image format extension"""
    from urllib.parse import urlparse

    from nim_llm_sdk.trtllm.llm_api_utils import LLMAPIValueError

    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    try:
        parsed = urlparse(url)

        # Allow base64 data URLs (validated downstream by the image loader)
        if parsed.scheme == 'data':
            return

        # Allow local files and bare paths without extension check; the loader will validate
        if parsed.scheme in ('file', ''):
            return

        # For http/https URLs, require a supported extension
        if parsed.scheme in ('http', 'https'):
            path = parsed.path.lower()
            if not any(path.endswith(ext) for ext in supported_extensions):
                raise LLMAPIValueError(
                    f"Unsupported image format in URL: {url}. Supported formats: JPEG, PNG, BMP, TIFF, WEBP"
                )
            return

        # For any other scheme, skip strict validation here
        return
    except LLMAPIValueError:
        raise  # Re-raise LLMAPIValueError as-is
    except Exception:
        # If URL parsing fails, let it continue (might be base64 or other format)
        pass

def validate_chat_request_images(request) -> None:
    """Validate all image URLs in a chat completion request"""
    if hasattr(request, 'messages') and request.messages:
        for message in request.messages:
            if hasattr(message, 'content') and isinstance(message.content, list):
                for content_part in message.content:
                    if (
                        isinstance(content_part, dict)
                        and content_part.get('type') == 'image_url'
                        and 'image_url' in content_part
                        and 'url' in content_part['image_url']
                    ):

                        image_url = content_part['image_url']['url']
                        print(f"🔍 Validating image URL: {image_url}")
                        validate_image_url_format(image_url)
                        print(f"✅ Image URL validated: {image_url}")


TIMEOUT_KEEP_ALIVE = 5  # seconds
# `envs._NIM_LICENSE_PATH` is not a constant but a call of `os.getenv`
# under the hood. The license path is reused in this module and thus
# the path is stored in a global variable.
NIM_LICENSE_PATH = envs._NIM_LICENSE_PATH
BASE_NGC_URL = "https://catalog.ngc.nvidia.com/orgs/nim/teams"

warm_up_completed = False

# Configure logger for __main__ module. Otherwise the default
# Python logging format will be used by `logger` object in this script.
configure_logger("")
configure_all_loggers_with_handlers_except(
    not_configured_loggers=["nim_llm_sdk", "uvicorn", ""], keep_original_log_level=True
)
logger = init_logger(__name__)

WARM_UP_HEALTH_CHECK_MESSAGE = "Service is starting"
EXAMPLE_COMPLETIONS_CURL_REQUEST = """curl -X 'POST' \\
  'http://{host}:{port}/v1/completions' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "{model}",
    "prompt": "hello world!",
    "top_p": 1,
    "n": 1,
    "max_tokens": 15,
    "stream": true,
    "frequency_penalty": 1.0,
    "stop": ["hello"]
  }}'
"""

EXAMPLE_CHAT_COMPLETIONS_CURL_REQUEST = """curl -X 'POST' \\
  'http://{host}:{port}/v1/chat/completions' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "{model}",
    "messages": [
      {{
        "role":"user",
        "content":"Hello! How are you?"
      }},
      {{
        "role":"assistant",
        "content":"Hi! I am quite well, how can I help you today?"
      }},
      {{
        "role":"user",
        "content":"Can you write me a song?"
      }}
    ],
    "top_p": 1,
    "n": 1,
    "max_tokens": 15,
    "stream": true,
    "frequency_penalty": 1.0,
    "stop": ["hello"]
  }}'
"""

EXAMPLE_VLM_CHAT_COMPLETIONS_CURL_REQUEST = """curl -X 'POST' \\
  'http://{host}:{port}/v1/chat/completions' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "{model}",
    "messages": [
      {{
        "role": "user",
        "content": [
          {{
            "type": "image_url",
            "image_url": {{
              "url": "https://upload.wikimedia.org/wikipedia/commons/c/c3/LibreOffice_Writer_6.3.png"
            }}
          }}
        ]
      }}
    ],
    "max_tokens": 256
  }}'
"""


def setup_metrics_registry():
    prometheus_multi_proc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multi_proc_dir is not None and os.path.isdir(prometheus_multi_proc_dir):
        from prometheus_client import CollectorRegistry, multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    else:
        return REGISTRY


async def serve_http(app: FastAPI, custom_signal_handler=True, **uvicorn_kwargs: Any):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ', '.join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    if custom_signal_handler:
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        if custom_signal_handler:
            return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.debug(
                "port %s is used by process %s launched with command:\n%s", port, process, " ".join(process.cmdline())
            )
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


class HealthCheck:
    def __init__(self, engine, is_vlm: bool):
        self.engine = engine
        self.successful = False
        self.is_vlm = is_vlm
        self.config = {
            "prompt": "health check",
            "stream": False,
            "temperature": 0.0,
            "request_id": str(uuid.uuid4()),
        }

    async def generate(self):
        sampling_params = SamplingParams(temperature=self.config["temperature"])
        if not self.is_vlm:
            generator = self.engine.generate(self.config["prompt"], sampling_params, "health-request")
        else:
            model_config = await self.engine.get_model_config()
            tokenizer = await self.engine.get_tokenizer()
            mm_tracker = MultiModalItemTracker(model_config, tokenizer)
            placeholder_str = mm_tracker.add("image", 1)

            image = np.random.randint(low=0, high=256, size=(128, 128, 3), dtype=np.uint8)
            image = Image.fromarray(image, 'RGB')
            inputs = {"prompt": f"health check\n{placeholder_str}\n", "multi_modal_data": {"image": image}}
            generator = self.engine.generate(inputs, sampling_params, "health-request")
        try:
            async for request_output in generator:
                if request_output.finished:
                    self.successful = True
                    return True
        except Exception as e:
            logger.exception(f"Error during health check: {e}")
            return False


def create_lifespan(
    engine,
    engine_args,
    is_vlm: bool,
    on_warmup_complete: Optional[Callable[[], None]] = None,
    stats_logging_period: int = 10,
):
    @asynccontextmanager
    async def lifespan():
        async def _force_log():
            while True:
                await asyncio.sleep(stats_logging_period)
                await engine.do_log_stats()

        async def _warm_up():
            warmup_checker = HealthCheck(engine, is_vlm)
            try:
                await warmup_checker.generate()
                if warmup_checker.successful:
                    global warm_up_completed
                    warm_up_completed = True
                    # Call the callback function if provided
                    if on_warmup_complete is not None:
                        on_warmup_complete()
            except asyncio.CancelledError as e:
                raise e
            except Exception as e:
                logger.exception("Error in warm up request during service start up")

        tasks = [asyncio.create_task(_warm_up())]

        if not engine_args.disable_log_stats:
            tasks.append(asyncio.create_task(_force_log()))

        gc.collect()
        gc.freeze()

        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return lifespan


async def create_streaming_response(
    gen: AsyncGenerator[str, None], openai_serving_chat: OpenAIServingChat
) -> StreamingResponse:
    try:
        first_r = await anext(gen)

        async def resp_stream():
            yield first_r
            async for r in gen:
                yield r

        return StreamingResponse(content=resp_stream(), media_type="text/event-stream")
    except LoraCacheFullError as e:
        err_resp = openai_serving_chat.create_streaming_error_response(
            str(e), "Too Many Requests", HTTPStatus.TOO_MANY_REQUESTS
        )

        async def resp_stream():
            yield f"data: {err_resp}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            content=resp_stream(),
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
            media_type="text/event-stream",
        )


def get_health_success_response(message: str):
    return JSONResponse(
        NIMHealthSuccessResponse(message=message).model_dump(),
        headers={"System-Health": str(True)},
        status_code=200,
    )


class ContentTypeException(Exception):
    pass


def verify_headers(content_type: Optional[str] = Header(None)):
    logger.debug("Verifying Content Type")
    if content_type != "application/json":
        raise ContentTypeException(f"Unsupported media type: {content_type}. It must be application/json")


def _create_error_response(original: ErrorResponse, *, status_code: Optional[int] = None) -> JSONResponse:
    content = original.model_dump()

    if envs._NIM_API_COMPAT_SINGLE_ERROR_FIELD:
        content = {"error": content["message"]}

    if status_code is None:
        status_code = original.code

    return JSONResponse(content=content, status_code=status_code)


def log_served_endpoints(app: fastapi.FastAPI, host: str | None, port: int) -> None:
    message = "Serving endpoints:\n  "
    endpoints_str = "\n  ".join([f"{'0.0.0.0' if host is None else host}:{port}{route.path}" for route in app.routes])
    logger.info(message + endpoints_str)


def log_example_curl_request(model: str, host: str | None, port: int, tokenizer: str, is_vlm: bool) -> None:
    """
    Models can support CHAT | COMPLETION | REWARD.
    If tokenizer file contains a chat template, log a chat completion sample request as model can perform one of CHAT ! REWARD tasks.
    Otherwise, log a completion sample request.
    """
    if _has_default_chat_template(tokenizer):
        EXAMPLE_CURL_REQUEST = (
            EXAMPLE_CHAT_COMPLETIONS_CURL_REQUEST if not is_vlm else EXAMPLE_VLM_CHAT_COMPLETIONS_CURL_REQUEST
        )
    else:
        EXAMPLE_CURL_REQUEST = EXAMPLE_COMPLETIONS_CURL_REQUEST
    logger.info(
        f"An example cURL request:\n"
        + EXAMPLE_CURL_REQUEST.format(model=model, host="0.0.0.0" if host is None else host, port=port),
    )


def _get_sha1(file_path):
    command = f"sha1sum {file_path}"
    result = os.popen(command).read()
    sha1 = result.split()[0]
    return sha1


def _get_model_info(model_dump: Dict) -> List[NIMModelInfoResponse]:
    model_info = []
    for model in model_dump['data']:
        model_id = model['id']
        try:
            org, model_name = model_id.split('/')
            model_url = f"{BASE_NGC_URL}/{org}/models/{model_name}"
        except ValueError as e:
            model_name = model_id
            model_url = ""
        model_info.append(NIMModelInfoResponse(shortName=model_name, modelUrl=model_url))
    return model_info


def _get_openapi_version():
    interface_instance = Interface()
    app = interface_instance.app
    openapi_schema = app.openapi()
    return openapi_schema.get("openapi", "unknown version")


def _get_license_info():
    # Currently, License URL is not implemented
    with open(NIM_LICENSE_PATH, "rb") as f:
        raw_content = f.read()
        content = raw_content.decode("utf-8", errors="replace")
    license_info = NIMLicenseInfoResponse(
        name=os.path.basename(NIM_LICENSE_PATH),
        path=NIM_LICENSE_PATH,
        sha=_get_sha1(NIM_LICENSE_PATH),
        size=os.path.getsize(NIM_LICENSE_PATH),
        url="",
        type="file",
        content=content,
    )
    return license_info


def _has_default_chat_template(tokenizer: str):
    tokenizer_config_path = os.path.join(tokenizer, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        tokenizer_config_path = None
    if tokenizer_config_path:
        with open(tokenizer_config_path, 'r') as f:
            data = json.load(f)
        return data.get("chat_template") is not None
    return False


class Interface(HttpNIMApiInterface):

    def __init__(self, *args, **kwargs):
        # self.engine = engine
        # self.engine_args = engine_args
        super().__init__(
            *args,
            title="NVIDIA NIM for LLMs",
            version=nim_llm_sdk.__version__,
            summary="Accelerated LLM inference for NVIDIA GPUs.",
            # TODO: pull in model card if it exists?
            description="",
            redoc_url=None,
            **kwargs,
        )
        # self.warm_up_completed = warm_up_completed
        # self.engine = engine

        self.openai_serving_chat: OpenAIServingChat = kwargs.get('openai_serving_chat', None)
        self.openai_serving_completion: OpenAIServingCompletion = kwargs.get('openai_serving_completion', None)
        self.llama_stack_serving_chat: LlamaStackServingChat = kwargs.get('llama_stack_serving_chat', None)
        self.llama_stack_serving_completion: LlamaStackServingCompletion = kwargs.get(
            'llama_stack_serving_completion', None
        )
        self.openai_serving_models: OpenAIServingModels = kwargs.get('openai_serving_models', None)

        self._registry = kwargs.get("registry", REGISTRY)
        self._metrics_initialized = False

    @HttpNIMApiInterface.route(
        "/v1/health/ready",
        methods=["get"],
        response_model=NIMHealthSuccessResponse,
        summary="Service ready check",
        description=dedent(
            """
        This endpoint will return a 200 status when the
        service is ready to receive inference requests
    """
        )
        .replace("\n", " ")
        .strip(),
        responses={
            503: {
                "description": "Service is not ready to receive requests.",
                "model": ErrorResponse,
            },
        },
    )
    async def health(self) -> Response:
        """Health check.
        Ensures all backend services are ready to serve inference requests.
        """
        if not warm_up_completed:
            err = self.openai_serving_chat.create_error_response(
                message=WARM_UP_HEALTH_CHECK_MESSAGE,
                err_type="ServiceUnavailableError",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return JSONResponse(err.model_dump(), status_code=HTTPStatus.SERVICE_UNAVAILABLE)
        try:
            # TODO: cleanup
            if isinstance(self.openai_serving_chat, MagicMock):
                await self.openai_serving_chat.engine.check_health()
            else:
                await self.openai_serving_chat.engine_client.check_health()
        except Exception as e:
            self.logger.exception("Error while checking service health")
            err = self.openai_serving_chat.create_error_response(
                message="Service in unhealthy",
                err_type="ServiceUnavailableError",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return JSONResponse(err.model_dump(), status_code=HTTPStatus.SERVICE_UNAVAILABLE)
        return get_health_success_response("Service is ready.")

    @HttpNIMApiInterface.route(
        "/v1/health/live",
        methods=["get"],
        summary="Service liveness check",
        description=dedent(
            """
                This endpoint will return a 200 status as soon as the service
                is able to accept traffic, but it does not mean
                the service is ready to accept inference requests.
            """
        )
        .replace("\n", " ")
        .strip(),
        response_model=Union[NIMHealthSuccessResponse],
    )
    async def health_ready(self) -> Response:
        return get_health_success_response("Service is live.")

    @HttpNIMApiInterface.route(
        "/v1/models",
        methods=["get"],
        summary="List available models",
        description=dedent(
            """
                This endpoint will return a list of models available for inference.
                When the NIM is set up to serve customizations (e.g. LoRAs) this will also return the
                customizations available as models.
            """
        )
        .replace("\n", " ")
        .strip(),
        response_model=ModelList,
    )
    async def show_available_models(self):
        models = await self.openai_serving_models.show_available_models()
        model_dump = models.model_dump()

        # For now, we always set the "owned_by" field to "system" (it is hard-coded to "vllm" in
        # the ModelCard class, so we have to overwrite it here). In the future, we may add additional
        # information to show the provenance of the model.

        for model in model_dump["data"]:
            model["owned_by"] = "system"

        return JSONResponse(content=model_dump)

    @HttpNIMApiInterface.route(
        "/v1/metadata",
        methods=["get"],
        summary="Provide NIM Metadata",
        description=dedent(
            """
                This endpoint will return the NIM Container Version, associated models, assets and license information
            """
        )
        .replace("\n", " ")
        .strip(),
        response_model=NIMMetadataResponse,
    )
    async def show_metadata(self):
        release_version = nim_llm_sdk.__version__
        api_version = _get_openapi_version()
        ver = NIMLLMVersionResponse(release=release_version, api=api_version)
        license_info = _get_license_info()
        models = await self.openai_serving_models.show_available_models()
        model_dump = models.model_dump()
        model_info = _get_model_info(model_dump)

        # For now asset info is empty. In the future we may provide some additional asset info here
        metadata = NIMMetadataResponse(version=ver, modelInfo=model_info, licenseInfo=license_info)
        return JSONResponse(content=metadata.model_dump())

    @HttpNIMApiInterface.route(
        "/v1/version",
        methods=["get"],
        summary="Returns version information about this NIM",
        description=dedent(
            """
                The `release` attribute corresponds to the product
                release version of the NIM. The `api` attribute
                is the openapi server API version running inside the NIM.
            """
        )
        .replace("\n", " ")
        .strip(),
        response_model=NIMLLMVersionResponse,
    )
    async def show_version(self):
        release_version = nim_llm_sdk.__version__
        api_version = _get_openapi_version()
        ver = NIMLLMVersionResponse(release=release_version, api=api_version)
        return JSONResponse(content=ver.model_dump())

    @HttpNIMApiInterface.route(
        "/v1/chat/completions",
        methods=["post"],
        summary="OpenAI-compatible chat endpoint",
        response_model=Union[ChatCompletionResponse, ChatCompletionStreamResponse],
        responses={
            400: {
                "description": "Received an invalid request possibly containing unsupported or out-of-range parameter values.",
                "model": ErrorResponse,
            },
            404: {
                "description": "The requested model does not exist.",
                "model": ErrorResponse,
            },
            500: {"description": "", "model": ErrorResponse},
            415: {"description": "Unsupported Media Type", "model": ErrorResponse},
        },
        dependencies=[Depends(verify_headers)],
    )
    async def create_chat_completion(self, request: NIMLLMChatCompletionRequest, raw_request: Request):

        model = request.model
        request = apply_custom_model_parameters(self.openai_serving_models.peft_provider, request)
        await maybe_set_custom_parameters(request)

        # Check if the model is supported to avoid raising ValueError in _maybe_get_adapters
        generator = await self.openai_serving_chat._check_model(request)
        if isinstance(generator, ErrorResponse):
            return _create_error_response(generator, status_code=generator.code)

        chat_template = request.chat_template or self.openai_serving_chat.chat_template
        lora_request = self.openai_serving_chat._maybe_get_adapters(request)
        tokenizer = await self.openai_serving_chat.engine_client.get_tokenizer(lora_request)
        is_mistral_tokenizer = isinstance(tokenizer, MistralTokenizer)
        # only HF tokenizer requires default chat template
        # vLLM already catches this but error message is enhanced here
        if not is_mistral_tokenizer:
            if chat_template is None and tokenizer.chat_template is None:
                raise ValueError(
                    f"Model {model} does not have a default chat template defined in the tokenizer.  Set the `chat_template` field to issue a `/chat/completions` request or use `/completions` endpoint"
                )

        # Validate image URLs in the request
        validate_chat_request_images(request)

        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
        generator = await maybe_update_constrained_response(request, generator, raw_request)
        if isinstance(generator, ErrorResponse):
            return _create_error_response(generator, status_code=generator.code)
        if request.stream:
            with nvtx.annotate("StreamingResponse", color="green", category="call"):
                ret = StreamingResponse(content=generator, media_type="text/event-stream")
            return ret
        else:
            assert isinstance(generator, ChatCompletionResponse)
            generator.model = model
            with nvtx.annotate("JSONResponse", color="green", category="call"):
                ret = JSONResponse(content=generator.model_dump())
            return ret

    @HttpNIMApiInterface.route(
        "/v1/completions",
        methods=["post"],
        summary="OpenAI-compatible completions endpoint",
        response_model=Union[CompletionResponse, CompletionStreamResponse],
        responses={
            400: {
                "description": "Received an invalid request possibly containing unsupported or out-of-range parameter values.",
                "model": ErrorResponse,
            },
            404: {
                "description": "The requested model does not exist.",
                "model": ErrorResponse,
            },
            500: {"description": "", "model": ErrorResponse},
            415: {"description": "Unsupported Media Type", "model": ErrorResponse},
        },
        dependencies=[Depends(verify_headers)],
    )
    async def create_completion(self, request: NIMLLMCompletionRequest, raw_request: Request):

        model = request.model
        request = apply_custom_model_parameters(self.openai_serving_models.peft_provider, request)

        if request.suffix is not None:
            suffix = request.suffix
            request.suffix = None

            # TODO: https://jirasw.nvidia.com/browse/INFE-3557
            basename_to_config = {
                # https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
                "deepseek": {"prefix": "<|fim_begin|>", "suffix": "<|fim_hole|>", "middle": "<|fim_end|>"},
                # https://huggingface.co/docs/transformers/en/model_doc/code_llama#transformers.CodeLlamaTokenizer
                "codellama": {"prefix": "_<PRE>", "suffix": "_<SUF>", "middle": "_<MID>"},
                # https://huggingface.co/bigcode/starcoder
                "starcoder": {"prefix": "<fim_prefix>", "suffix": "<fim_suffix>", "middle": "<fim_middle>"},
                # https://huggingface.co/google/codegemma-7b
                "codegemma": {"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"},
            }

            success = False
            for basename, config in basename_to_config.items():
                # TODO: https://jirasw.nvidia.com/browse/INFE-3556
                if basename in request.model:
                    prefix_token = config["prefix"]
                    suffix_token = config["suffix"]
                    middle_token = config["middle"]
                    request.prompt = f"{prefix_token}{request.prompt}{suffix_token}{suffix}{middle_token}"
                    success = True
                    break

            if not success:
                request.suffix = suffix
                self.logger.warning("Unsupported model name: %s for suffix", request.model)

        await maybe_set_custom_parameters(request)
        generator = await self.openai_serving_completion.create_completion(request, raw_request)
        generator = await maybe_update_constrained_response(request, generator, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            with nvtx.annotate("StreamingResponse", color="green", category="call"):
                ret = StreamingResponse(content=generator, media_type="text/event-stream")
            return ret
        else:
            generator.model = model
            with nvtx.annotate("JSONResponse", color="green", category="call"):
                ret = JSONResponse(content=generator.model_dump())
            return ret

    @HttpNIMApiInterface.route(
        "/experimental/ls/inference/chat_completion",
        methods=["post"],
        summary="Llama Stack-compatible chat endpoint",
        response_model=Union[LlamaStackChatCompletionResponse, LlamaStackChatCompletionResponseStreamChunk],
        responses={
            400: {
                "description": "Received an invalid request possibly containing unsupported or out-of-range parameter values.",
                "model": ErrorResponse,
            },
            404: {
                "description": "The requested model does not exist.",
                "model": ErrorResponse,
            },
            500: {"description": "", "model": ErrorResponse},
            415: {"description": "Unsupported Media Type", "model": ErrorResponse},
        },
        dependencies=[Depends(verify_headers)],
    )
    async def create_llama_stack_chat_completion(self, request: LlamaStackChatCompletionRequest, raw_request: Request):
        generator = await self.llama_stack_serving_chat.create_chat_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return _create_error_response(generator)
        if request.stream:
            return await create_streaming_response(generator, self.openai_serving_chat)
        else:
            assert isinstance(generator, LlamaStackChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    @HttpNIMApiInterface.route(
        "/experimental/ls/inference/completion",
        methods=["post"],
        summary="LlamaStack-compatible completion endpoint",
        response_model=Union[LlamaStackCompletionResponse, LlamaStackCompletionResponseStreamChunk],
        responses={
            400: {
                "description": "Received an invalid request possibly containing unsupported or out-of-range parameter values.",
                "model": ErrorResponse,
            },
            404: {
                "description": "The requested model does not exist.",
                "model": ErrorResponse,
            },
            500: {"description": "", "model": ErrorResponse},
            415: {"description": "Unsupported Media Type", "model": ErrorResponse},
        },
        dependencies=[Depends(verify_headers)],
    )
    async def create_llama_stack_completion(self, request: LlamaStackCompletionRequest, raw_request: Request):
        generator = await self.llama_stack_serving_completion.create_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return _create_error_response(generator)
        if request.stream:
            return await create_streaming_response(generator, self.openai_serving_chat)
        else:
            assert isinstance(generator, LlamaStackCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    @HttpNIMApiInterface.exception_handler(RequestValidationError)
    async def validation_exception_handler(self, request: Request, exc: RequestValidationError):
        status = HTTPStatus.BAD_REQUEST
        err_type = "BadRequestError"
        err = self.openai_serving_chat.create_error_response(message=str(exc), status_code=status, err_type=err_type)
        return _create_error_response(err, status_code=status)

    @HttpNIMApiInterface.exception_handler(LoraCacheFullError)
    async def lora_cache_full_handler(self, request: Request, exc: LoraCacheFullError):
        status = HTTPStatus.TOO_MANY_REQUESTS
        err_type = "RateLimitError"
        err = self.openai_serving_chat.create_error_response(message=str(exc), status_code=status, err_type=err_type)
        return _create_error_response(err, status_code=status)

    @HttpNIMApiInterface.exception_handler(ContentTypeException)
    async def incorrect_content_type_exception_handler(self, request: Request, exc: ContentTypeException):
        status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
        err_type = "BadRequestError"
        err = self.openai_serving_chat.create_error_response(message=str(exc), status_code=status, err_type=err_type)
        return _create_error_response(err, status_code=status)

    @HttpNIMApiInterface.exception_handler(LLMAPIValueError)
    async def llm_api_value_error_handler(self, request: Request, exc: LLMAPIValueError):
        status = HTTPStatus.BAD_REQUEST
        err_type = "BadRequestError"
        err = self.openai_serving_chat.create_error_response(message=str(exc), status_code=status, err_type=err_type)
        return _create_error_response(err, status_code=status)

    @HttpNIMApiInterface.exception_handler(Exception)
    async def fallback_exception_handler(self, request: Request, exc: Exception):
        status = HTTPStatus.INTERNAL_SERVER_ERROR
        err_type = "InternalServerError"
        err = self.openai_serving_chat.create_error_response(message=str(exc), status_code=status, err_type=err_type)
        return _create_error_response(err, status_code=status)

    @override
    async def metrics(self) -> str:
        from prometheus_client import generate_latest

        content = generate_latest(registry=self._registry)
        return StarletteResponse(content=content, media_type="text/plain")


async def main():
    logger.info(f"NIM LLM API version {nim_llm_sdk.__version__}")
    comm = MPI.COMM_WORLD
    mpi_world_size = comm.size
    mpi_rank = comm.rank
    # this environment variable is OpenMPI specific
    mpi_local_rank: int = envs.OMPI_COMM_WORLD_LOCAL_RANK
    logger.debug(f"MPI rank: {mpi_rank}")
    logger.debug(f"MPI world size: {mpi_world_size}")
    logger.debug(f"MPI local node rank: {mpi_local_rank}")

    if mpi_rank == 0:
        inference_env = prepare_environment()
        served_model_names = inference_env.served_model_name
        base_model_paths = [BaseModelPath(name=name, model_path=name) for name in served_model_names]
        args = inference_env.parsed_args
        engine_args = inference_env.engine_args
        engine_args.max_log_len = args.max_log_len
        model_name = args.model
        engine_world_size = engine_args.tensor_parallel_size * inference_env.engine_args.pipeline_parallel_size

    else:
        engine_args = None
        model_name = None

    if mpi_world_size > 1:
        engine_args: NimAsyncEngineArgs = comm.bcast(engine_args, root=0)
        model_name: str = comm.bcast(model_name, root=0)

    # each node will need to create the model repo
    # the first node has already done this above
    if mpi_rank != 0 and mpi_local_rank == 0:
        model_repo = engine_args.model
        engine_args.model = model_name
        engine_args, _ = inject_ngc_hub(engine_args, model_repo_target_dir=model_repo)

    # wait for model repos to be created on each node
    comm.barrier()

    use_trtllm = is_trt_llm_model(engine_args)

    if use_trtllm:
        from nim_llm_sdk.trtllm.utils import get_device_ids_from_global_world_size

        selected_gpus = deepcopy(engine_args.selected_gpus)
        logger.info(f"the number of selected_gpus is {len(selected_gpus)}")
        device_ids = get_device_ids_from_global_world_size(selected_gpus, mpi_world_size)
        logger.info(f"mpi_local_rank is {mpi_local_rank}")
        logger.info(f"mpi_world_size is {mpi_world_size}")
        torch.cuda.set_device(device_ids[mpi_local_rank])

    jit_config = None
    try:
        if mpi_rank == 0:
            # sd=dasda
            # Register any custom tool call parsers here
            tool_parser_plugin = None
            if hasattr(inference_env.parsed_args, "tool_parser_plugin"):
                tool_parser_plugin = inference_env.parsed_args.tool_parser_plugin
            if tool_parser_plugin is not None and len(tool_parser_plugin) > len(".py"):
                logger.info(f"Attempting to register tool parser plugin: {tool_parser_plugin}")
                from vllm.entrypoints.openai.tool_parsers import ToolParserManager

                try:
                    ToolParserManager.import_tool_parser(tool_parser_plugin)
                except Exception as e:
                    logger.error(f"Failed to register tool parser plugin: {e}")
                    raise e

            async with AsyncLLMEngineFactory.from_engine_args(
                engine_args, usage_context=UsageContext.OPENAI_API_SERVER
            ) as engine:
                jit_config = engine.jit_config if hasattr(engine, "jit_config") else None

                model_config = await engine.get_model_config()

                if args.disable_log_requests:
                    request_logger = None
                else:
                    request_logger = RequestLogger(max_log_len=args.max_log_len)

                synchronizer = init_model_synchronizers(
                    served_model_names, args.enable_lora, args.peft_source, args.peft_refresh_interval
                )

                openai_serving_models = OpenAIServingModels(
                    engine_client=engine,
                    model_config=model_config,
                    base_model_paths=base_model_paths,
                    lora_modules=args.lora_modules,
                    synchronizer=synchronizer,
                )
                await openai_serving_models.init_static_loras()

                resolved_chat_template = load_chat_template(args.chat_template)
                if resolved_chat_template is not None:
                    logger.info("Using supplied chat template:\n%s", resolved_chat_template)
                openai_serving_chat = OpenAIServingChat(
                    engine_client=engine,
                    model_config=model_config,
                    models=openai_serving_models,
                    response_role=args.response_role,
                    request_logger=request_logger,
                    chat_template=resolved_chat_template,
                    chat_template_content_format=args.chat_template_content_format,
                    return_tokens_as_token_ids=args.return_tokens_as_token_ids,
                    enable_auto_tools=args.enable_auto_tool_choice,
                    tool_parser=args.tool_call_parser,
                    reasoning_parser=args.reasoning_parser,
                    enable_prompt_tokens_details=args.enable_prompt_tokens_details,
                )
                openai_serving_completion = OpenAIServingCompletion(
                    engine_client=engine,
                    model_config=model_config,
                    models=openai_serving_models,
                    request_logger=request_logger,
                    return_tokens_as_token_ids=args.return_tokens_as_token_ids,
                )

                llama_stack_serving_chat = LlamaStackServingChat(
                    engine_client=engine,
                    model_config=model_config,
                    models=openai_serving_models,
                    response_role=args.response_role,
                    request_logger=request_logger,
                    chat_template=resolved_chat_template,
                    chat_template_content_format=args.chat_template_content_format,
                    return_tokens_as_token_ids=args.return_tokens_as_token_ids,
                    enable_auto_tools=args.enable_auto_tool_choice,
                    tool_parser=args.tool_call_parser,
                    reasoning_parser=args.reasoning_parser,
                    enable_prompt_tokens_details=args.enable_prompt_tokens_details,
                )
                llama_stack_serving_completion = LlamaStackServingCompletion(
                    engine_client=engine,
                    model_config=model_config,
                    models=openai_serving_models,
                    request_logger=request_logger,
                    return_tokens_as_token_ids=args.return_tokens_as_token_ids,
                )

                is_vlm = model_config.is_multimodal_model or hasattr(model_config.hf_config, "vision_config")

                cache_model(jit_config, engine_args)

                # Create a function to log example curl request
                def log_example_curl_after_warmup():
                    log_example_curl_request(
                        served_model_names[0], args.host, args.port, model_config.tokenizer, is_vlm
                    )

                registry = setup_metrics_registry()
                app_interface = Interface(
                    lifespan=create_lifespan(
                        engine=engine,
                        engine_args=engine_args,
                        is_vlm=is_vlm,
                        on_warmup_complete=log_example_curl_after_warmup,
                    ),
                    middleware=args.middleware,
                    allow_origins=args.allowed_origins,
                    allow_credentials=args.allow_credentials,
                    allow_methods=args.allowed_methods,
                    allow_headers=args.allowed_headers,
                    openai_serving_models=openai_serving_models,
                    openai_serving_chat=openai_serving_chat,
                    openai_serving_completion=openai_serving_completion,
                    llama_stack_serving_chat=llama_stack_serving_chat,
                    llama_stack_serving_completion=llama_stack_serving_completion,
                    registry=registry,
                )
                my_app = app_interface.app

                set_prompt_telemetry_webhooks(envs.NIM_PROMPT_TELEMETRY_EXPORT_WEBHOOKS)

                # Add request context middleware first, to ensure it runs after the request body transformation middleware
                from nim_llm_sdk.entrypoints.openai.middleware.set_request_context import RequestContextMiddleware

                app_interface.app.add_middleware(RequestContextMiddleware)

                if is_prompt_telemetry_enabled():
                    app_interface.app.add_middleware(PromptTelemetryMiddleware)

                # Add request body transformation middleware last, to ensure it runs first upon receiving the request
                if envs.NIM_USE_API_TRANSFORM_SHIM:
                    app_interface.app.add_middleware(FieldTransformationMiddleware)

                shutdown_task = await serve_http(
                    my_app,
                    custom_signal_handler=(not use_trtllm),
                    host=args.host,
                    port=args.port,
                    log_level=args.uvicorn_log_level,
                    workers=1,
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=args.ssl_keyfile,
                    ssl_certfile=args.ssl_certfile,
                    ssl_ca_certs=args.ssl_ca_certs,
                    ssl_cert_reqs=args.ssl_cert_reqs,
                    log_config=get_logging_config_for_package("uvicorn"),
                )

                await shutdown_task

        else:
            from nim_llm_sdk.trtllm.weight_manager.build_utils import JITTrtllmConfig

            jit_config = JITTrtllmConfig.from_engine_args(engine_args)
            lora_config = extract_lora_config(engine_args)
            engine_config = engine_args.create_engine_config()
            # worker ranks will block
            from nim_llm_sdk.trtllm.utils import create_trtllm_executor, create_vision_processor

            create_vision_processor(engine_config.model_config, None, jit_config)

            trt_llm_engine = create_trtllm_executor(
                engine_config.model_config,
                engine_config.parallel_config,
                engine_config.scheduler_config,
                engine_config.device_config,
                engine_config.cache_config,
                lora_config,
                log_stats=not engine_args.disable_log_stats,
                selected_gpus=engine_args.selected_gpus,
                force_leader_mode=True,
                jit_config=jit_config,
            )

    except Exception:
        if mpi_world_size > 1:
            logger.exception("Critical error in rank %s, aborting MPI", mpi_rank)
            comm.Abort(1)
            # Abort should never return, but raise as fallback safety measure
            raise
        else:
            logger.exception("Critical error, exiting")
            raise

    # ensure non-rank0s don't exit early
    if mpi_world_size > 1:
        comm.barrier()


if __name__ == "__main__":
    uvloop.run(main())
