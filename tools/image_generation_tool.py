#!/usr/bin/env python3
"""
Image Generation Tools Module

This module provides image generation tools using FAL.ai's FLUX 2 Pro model with 
automatic upscaling via FAL.ai's Clarity Upscaler for enhanced image quality.

Available tools:
- image_generate_tool: Generate images from text prompts with automatic upscaling

Features:
- High-quality image generation using FLUX 2 Pro model
- Automatic 2x upscaling using Clarity Upscaler for enhanced quality
- Comprehensive parameter control (size, steps, guidance, etc.)
- Proper error handling and validation with fallback to original images
- Debug logging support
- Sync mode for immediate results

Usage:
    from image_generation_tool import image_generate_tool
    import asyncio
    
    # Generate and automatically upscale an image
    result = await image_generate_tool(
        prompt="A serene mountain landscape with cherry blossoms",
        image_size="landscape_4_3",
        num_images=1
    )
"""

import base64
import json
import logging
import mimetypes
import os
import re
import datetime
import threading
import uuid
from typing import Dict, Any, Optional, Union
from urllib.parse import urlencode
from pathlib import Path
import fal_client
from hermes_constants import OPENROUTER_BASE_URL
from tools.debug_helpers import DebugSession
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import managed_nous_tools_enabled

logger = logging.getLogger(__name__)

# Configuration for image generation
DEFAULT_MODEL = "fal-ai/flux-2-pro"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-image-preview"
DEFAULT_ASPECT_RATIO = "landscape"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_NUM_IMAGES = 1
DEFAULT_OUTPUT_FORMAT = "png"

IMAGE_PROVIDER_FAL = "fal"
IMAGE_PROVIDER_OPENROUTER = "openrouter"
_SUPPORTED_IMAGE_PROVIDERS = frozenset({IMAGE_PROVIDER_FAL, IMAGE_PROVIDER_OPENROUTER})

SUPPORTED_OPENROUTER_IMAGE_MODELS = frozenset({
    "google/gemini-2.5-flash-image",
    "google/gemini-3.1-flash-image-preview",
    "openai/gpt-5-image-mini",
})

_OPENROUTER_TEXT_AND_IMAGE_MODELS = frozenset({
    "google/gemini-2.5-flash-image",
    "google/gemini-3.1-flash-image-preview",
    "openai/gpt-5-image-mini",
})

_OPENROUTER_IMAGE_CONFIG_ASPECT_RATIO_MODELS = frozenset({
    "google/gemini-2.5-flash-image",
    "google/gemini-3.1-flash-image-preview",
})

_OPENROUTER_ASPECT_RATIO_MAP = {
    "square": "1:1",
    "portrait": "9:16",
    "landscape": "16:9",
}

# Safety settings
ENABLE_SAFETY_CHECKER = False
SAFETY_TOLERANCE = "5"  # Maximum tolerance (1-5, where 5 is most permissive)

# Aspect ratio mapping - simplified choices for model to select
ASPECT_RATIO_MAP = {
    "landscape": "landscape_16_9",
    "square": "square_hd",
    "portrait": "portrait_16_9"
}

# Configuration for automatic upscaling
UPSCALER_MODEL = "fal-ai/clarity-upscaler"
UPSCALER_FACTOR = 2
UPSCALER_SAFETY_CHECKER = False
UPSCALER_DEFAULT_PROMPT = "masterpiece, best quality, highres"
UPSCALER_NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2)"
UPSCALER_CREATIVITY = 0.35
UPSCALER_RESEMBLANCE = 0.6
UPSCALER_GUIDANCE_SCALE = 4
UPSCALER_NUM_INFERENCE_STEPS = 18

# Valid parameter values for validation based on FLUX 2 Pro documentation
VALID_IMAGE_SIZES = [
    "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
]
VALID_OUTPUT_FORMATS = ["jpeg", "png"]
VALID_ACCELERATION_MODES = ["none", "regular", "high"]

_debug = DebugSession("image_tools", env_var="IMAGE_TOOLS_DEBUG")
_managed_fal_client = None
_managed_fal_client_config = None
_managed_fal_client_lock = threading.Lock()


def _resolve_managed_fal_gateway():
    """Return managed fal-queue gateway config when direct FAL credentials are absent."""
    if os.getenv("FAL_KEY"):
        return None
    return resolve_managed_tool_gateway("fal-queue")


def _normalize_fal_queue_url_format(queue_run_origin: str) -> str:
    normalized_origin = str(queue_run_origin or "").strip().rstrip("/")
    if not normalized_origin:
        raise ValueError("Managed FAL queue origin is required")
    return f"{normalized_origin}/"


class _ManagedFalSyncClient:
    """Small per-instance wrapper around fal_client.SyncClient for managed queue hosts."""

    def __init__(self, *, key: str, queue_run_origin: str):
        sync_client_class = getattr(fal_client, "SyncClient", None)
        if sync_client_class is None:
            raise RuntimeError("fal_client.SyncClient is required for managed FAL gateway mode")

        client_module = getattr(fal_client, "client", None)
        if client_module is None:
            raise RuntimeError("fal_client.client is required for managed FAL gateway mode")

        self._queue_url_format = _normalize_fal_queue_url_format(queue_run_origin)
        self._sync_client = sync_client_class(key=key)
        self._http_client = getattr(self._sync_client, "_client", None)
        self._maybe_retry_request = getattr(client_module, "_maybe_retry_request", None)
        self._raise_for_status = getattr(client_module, "_raise_for_status", None)
        self._request_handle_class = getattr(client_module, "SyncRequestHandle", None)
        self._add_hint_header = getattr(client_module, "add_hint_header", None)
        self._add_priority_header = getattr(client_module, "add_priority_header", None)
        self._add_timeout_header = getattr(client_module, "add_timeout_header", None)

        if self._http_client is None:
            raise RuntimeError("fal_client.SyncClient._client is required for managed FAL gateway mode")
        if self._maybe_retry_request is None or self._raise_for_status is None:
            raise RuntimeError("fal_client.client request helpers are required for managed FAL gateway mode")
        if self._request_handle_class is None:
            raise RuntimeError("fal_client.client.SyncRequestHandle is required for managed FAL gateway mode")

    def submit(
        self,
        application: str,
        arguments: Dict[str, Any],
        *,
        path: str = "",
        hint: Optional[str] = None,
        webhook_url: Optional[str] = None,
        priority: Any = None,
        headers: Optional[Dict[str, str]] = None,
        start_timeout: Optional[Union[int, float]] = None,
    ):
        url = self._queue_url_format + application
        if path:
            url += "/" + path.lstrip("/")
        if webhook_url is not None:
            url += "?" + urlencode({"fal_webhook": webhook_url})

        request_headers = dict(headers or {})
        if hint is not None and self._add_hint_header is not None:
            self._add_hint_header(hint, request_headers)
        if priority is not None:
            if self._add_priority_header is None:
                raise RuntimeError("fal_client.client.add_priority_header is required for priority requests")
            self._add_priority_header(priority, request_headers)
        if start_timeout is not None:
            if self._add_timeout_header is None:
                raise RuntimeError("fal_client.client.add_timeout_header is required for timeout requests")
            self._add_timeout_header(start_timeout, request_headers)

        response = self._maybe_retry_request(
            self._http_client,
            "POST",
            url,
            json=arguments,
            timeout=getattr(self._sync_client, "default_timeout", 120.0),
            headers=request_headers,
        )
        self._raise_for_status(response)

        data = response.json()
        return self._request_handle_class(
            request_id=data["request_id"],
            response_url=data["response_url"],
            status_url=data["status_url"],
            cancel_url=data["cancel_url"],
            client=self._http_client,
        )


def _get_managed_fal_client(managed_gateway):
    """Reuse the managed FAL client so its internal httpx.Client is not leaked per call."""
    global _managed_fal_client, _managed_fal_client_config

    client_config = (
        managed_gateway.gateway_origin.rstrip("/"),
        managed_gateway.nous_user_token,
    )
    with _managed_fal_client_lock:
        if _managed_fal_client is not None and _managed_fal_client_config == client_config:
            return _managed_fal_client

        _managed_fal_client = _ManagedFalSyncClient(
            key=managed_gateway.nous_user_token,
            queue_run_origin=managed_gateway.gateway_origin,
        )
        _managed_fal_client_config = client_config
        return _managed_fal_client


def _submit_fal_request(model: str, arguments: Dict[str, Any]):
    """Submit a FAL request using direct credentials or the managed queue gateway."""
    request_headers = {"x-idempotency-key": str(uuid.uuid4())}
    managed_gateway = _resolve_managed_fal_gateway()
    if managed_gateway is None:
        return fal_client.submit(model, arguments=arguments, headers=request_headers)

    managed_client = _get_managed_fal_client(managed_gateway)
    return managed_client.submit(
        model,
        arguments=arguments,
        headers=request_headers,
    )


def _normalize_image_provider(provider: Any) -> str:
    normalized = str(provider or "").strip().lower()
    if not normalized:
        return IMAGE_PROVIDER_FAL
    if normalized in _SUPPORTED_IMAGE_PROVIDERS:
        return normalized
    raise ValueError(
        f"Unsupported image_generation.provider '{provider}'. "
        f"Supported providers: {sorted(_SUPPORTED_IMAGE_PROVIDERS)}"
    )


def _load_image_generation_config() -> Dict[str, Any]:
    config = {
        "provider": IMAGE_PROVIDER_FAL,
        "model": DEFAULT_MODEL,
        "base_url": "",
        "api_key": "",
        "timeout": 120,
        "provider_error": "",
    }
    try:
        from hermes_cli.config import load_config

        loaded = load_config().get("image_generation", {})
        if not isinstance(loaded, dict):
            return config

        provider = _normalize_image_provider(loaded.get("provider"))
        model = str(loaded.get("model") or "").strip()
        if not model:
            model = DEFAULT_OPENROUTER_MODEL if provider == IMAGE_PROVIDER_OPENROUTER else DEFAULT_MODEL

        timeout = loaded.get("timeout", config["timeout"])
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = config["timeout"]

        config.update({
            "provider": provider,
            "model": model,
            "base_url": str(loaded.get("base_url") or "").strip(),
            "api_key": str(loaded.get("api_key") or "").strip(),
            "timeout": timeout,
            "provider_error": "",
        })
    except Exception as exc:
        config["provider_error"] = str(exc)
    return config


def _openrouter_api_key_available(config: Optional[Dict[str, Any]] = None) -> bool:
    image_config = config or _load_image_generation_config()
    return bool(str(image_config.get("api_key") or "").strip() or os.getenv("OPENROUTER_API_KEY"))


def _get_openrouter_client(config: Optional[Dict[str, Any]] = None):
    image_config = config or _load_image_generation_config()
    explicit_base_url = str(image_config.get("base_url") or "").strip()
    explicit_api_key = str(image_config.get("api_key") or "").strip()

    if explicit_base_url or explicit_api_key:
        from openai import OpenAI

        api_key = explicit_api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        base_url = explicit_base_url or OPENROUTER_BASE_URL
        extra = {}
        if "openrouter.ai" in base_url.lower():
            try:
                from agent.auxiliary_client import _OR_HEADERS

                extra["default_headers"] = dict(_OR_HEADERS)
            except Exception:
                pass
        return OpenAI(api_key=api_key, base_url=base_url, **extra)

    from agent.auxiliary_client import resolve_provider_client

    client, _resolved_model = resolve_provider_client(
        IMAGE_PROVIDER_OPENROUTER,
        model=str(image_config.get("model") or DEFAULT_OPENROUTER_MODEL),
    )
    if client is None:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return client


def _openrouter_modalities_for_model(model: str) -> list[str]:
    if model in _OPENROUTER_TEXT_AND_IMAGE_MODELS:
        return ["image", "text"]
    return ["image"]


def _openrouter_image_config_for_request(model: str, aspect_ratio: str) -> Optional[Dict[str, Any]]:
    normalized_aspect_ratio = str(aspect_ratio or DEFAULT_ASPECT_RATIO).strip().lower()
    requested_ratio = _OPENROUTER_ASPECT_RATIO_MAP.get(normalized_aspect_ratio)
    if not requested_ratio:
        raise ValueError(f"Unsupported aspect_ratio '{aspect_ratio}' for OpenRouter image generation")

    if model in _OPENROUTER_IMAGE_CONFIG_ASPECT_RATIO_MODELS:
        return {"aspect_ratio": requested_ratio}

    if normalized_aspect_ratio == DEFAULT_ASPECT_RATIO:
        return None

    raise ValueError(
        f"OpenRouter model '{model}' does not support Hermes aspect_ratio control; "
        "choose a Gemini image model or use the default landscape setting"
    )


def _image_extension_from_media_type(media_type: str) -> str:
    normalized = str(media_type or "").strip().lower()
    if normalized == "image/jpeg":
        return ".jpg"
    if normalized in {"image/png", "image/webp", "image/gif", "image/bmp"}:
        return "." + normalized.rsplit("/", 1)[-1]
    return ".png"


def _image_media_type_from_path(path: str) -> str:
    guessed = (mimetypes.guess_type(path)[0] or "").lower()
    if guessed.startswith("image/"):
        return guessed

    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".bmp":
        return "image/bmp"
    raise ValueError(f"Unsupported input image type for '{path}'")


def _input_image_to_openrouter_url(image_ref: str) -> str:
    value = str(image_ref or "").strip()
    if not value:
        raise ValueError("Input image reference must be a non-empty string")
    if value.startswith("data:image/"):
        return value
    if value.startswith(("http://", "https://")):
        return value

    image_path = Path(value)
    if not image_path.exists():
        raise ValueError(f"Input image does not exist: {value}")
    if not image_path.is_file():
        raise ValueError(f"Input image path is not a file: {value}")

    media_type = _image_media_type_from_path(value)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _build_openrouter_messages(prompt: str, input_images: Optional[list[str]] = None) -> list[Dict[str, Any]]:
    prompt_text = prompt.strip()
    if not input_images:
        return [{"role": "user", "content": prompt_text}]

    content: list[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for image_ref in input_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _input_image_to_openrouter_url(image_ref)},
            }
        )
    return [{"role": "user", "content": content}]


def _materialize_image_reference(image_ref: str) -> str:
    image_ref = str(image_ref or "").strip()
    if not image_ref.startswith("data:"):
        return image_ref

    header, separator, encoded = image_ref.partition(",")
    if not separator:
        raise ValueError("Malformed image data URL returned by OpenRouter")
    if ";base64" not in header.lower():
        raise ValueError("Unsupported non-base64 image data URL returned by OpenRouter")

    media_type = "image/png"
    if ":" in header and ";" in header:
        media_type = header.split(":", 1)[1].split(";", 1)[0].strip() or media_type

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image data returned by OpenRouter") from exc

    from gateway.platforms.base import cache_image_from_bytes

    return cache_image_from_bytes(image_bytes, _image_extension_from_media_type(media_type))


def _extract_image_url_from_entry(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""

    image_url = entry.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url", "")
        if isinstance(url, str) and url.strip():
            return url.strip()
    elif isinstance(image_url, str) and image_url.strip():
        return image_url.strip()

    url = entry.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()

    b64_json = entry.get("b64_json")
    if isinstance(b64_json, str) and b64_json.strip():
        return f"data:image/png;base64,{b64_json.strip()}"

    return ""


def _extract_image_refs_from_content_string(content: str) -> list[str]:
    text = str(content or "").strip()
    if not text:
        return []

    refs: list[str] = []

    for match in re.finditer(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+", text):
        refs.append(match.group(0))

    markdown_pattern = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+|data:image/[^)\s]+)\)")
    for match in markdown_pattern.finditer(text):
        refs.append(match.group(1))

    url_pattern = re.compile(r"https?://\S+")
    for match in url_pattern.finditer(text):
        candidate = match.group(0).rstrip(').,]')
        refs.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref and ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def _extract_openrouter_image_references(payload: Dict[str, Any]) -> list[str]:
    image_refs: list[str] = []

    for choice in payload.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        images = message.get("images")
        if isinstance(images, list):
            for entry in images:
                image_ref = _extract_image_url_from_entry(entry)
                if image_ref:
                    image_refs.append(image_ref)

        content = message.get("content")
        if isinstance(content, list):
            for entry in content:
                image_ref = _extract_image_url_from_entry(entry)
                if image_ref:
                    image_refs.append(image_ref)
        elif isinstance(content, str):
            image_refs.extend(_extract_image_refs_from_content_string(content))

    data = payload.get("data")
    if isinstance(data, list):
        for entry in data:
            image_ref = _extract_image_url_from_entry(entry)
            if image_ref:
                image_refs.append(image_ref)

    deduped: list[str] = []
    seen: set[str] = set()
    for ref in image_refs:
        if ref and ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def _generate_via_openrouter(
    prompt: str,
    *,
    aspect_ratio: str,
    input_images: Optional[list[str]],
    image_config: Dict[str, Any],
    debug_call_data: Dict[str, Any],
    start_time: datetime.datetime,
) -> str:
    model = str(image_config.get("model") or DEFAULT_OPENROUTER_MODEL).strip()
    if model not in SUPPORTED_OPENROUTER_IMAGE_MODELS:
        raise ValueError(
            "Unsupported OpenRouter image model "
            f"'{model}'. Supported models: {sorted(SUPPORTED_OPENROUTER_IMAGE_MODELS)}"
        )

    request_image_config = _openrouter_image_config_for_request(model, aspect_ratio)
    client = _get_openrouter_client(image_config)
    create_kwargs = {
        "model": model,
        "messages": _build_openrouter_messages(prompt, input_images),
        "modalities": _openrouter_modalities_for_model(model),
        "stream": False,
        "timeout": image_config.get("timeout", 120),
    }
    if request_image_config:
        create_kwargs["extra_body"] = {"image_config": request_image_config}

    logger.info("Generating image with OpenRouter model: %s", model)
    result = client.chat.completions.create(**create_kwargs)
    payload = result.model_dump() if hasattr(result, "model_dump") else result
    if not isinstance(payload, dict):
        raise ValueError("Invalid response from OpenRouter image API")

    image_refs = _extract_openrouter_image_references(payload)
    if not image_refs:
        raise ValueError("OpenRouter returned no images")

    image_path_or_url = _materialize_image_reference(image_refs[0])
    generation_time = (datetime.datetime.now() - start_time).total_seconds()

    debug_call_data["success"] = True
    debug_call_data["images_generated"] = len(image_refs)
    debug_call_data["generation_time"] = generation_time
    _debug.log_call("image_generate_tool", debug_call_data)
    _debug.save()

    return json.dumps({"success": True, "image": image_path_or_url}, indent=2, ensure_ascii=False)


def _validate_parameters(
    image_size: Union[str, Dict[str, int]], 
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    output_format: str,
    acceleration: str = "none"
) -> Dict[str, Any]:
    """
    Validate and normalize image generation parameters for FLUX 2 Pro model.
    
    Args:
        image_size: Either a preset string or custom size dict
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale value
        num_images: Number of images to generate
        output_format: Output format for images
        acceleration: Acceleration mode for generation speed
    
    Returns:
        Dict[str, Any]: Validated and normalized parameters
    
    Raises:
        ValueError: If any parameter is invalid
    """
    validated = {}
    
    # Validate image_size
    if isinstance(image_size, str):
        if image_size not in VALID_IMAGE_SIZES:
            raise ValueError(f"Invalid image_size '{image_size}'. Must be one of: {VALID_IMAGE_SIZES}")
        validated["image_size"] = image_size
    elif isinstance(image_size, dict):
        if "width" not in image_size or "height" not in image_size:
            raise ValueError("Custom image_size must contain 'width' and 'height' keys")
        if not isinstance(image_size["width"], int) or not isinstance(image_size["height"], int):
            raise ValueError("Custom image_size width and height must be integers")
        if image_size["width"] < 64 or image_size["height"] < 64:
            raise ValueError("Custom image_size dimensions must be at least 64x64")
        if image_size["width"] > 2048 or image_size["height"] > 2048:
            raise ValueError("Custom image_size dimensions must not exceed 2048x2048")
        validated["image_size"] = image_size
    else:
        raise ValueError("image_size must be either a preset string or a dict with width/height")
    
    # Validate num_inference_steps
    if not isinstance(num_inference_steps, int) or num_inference_steps < 1 or num_inference_steps > 100:
        raise ValueError("num_inference_steps must be an integer between 1 and 100")
    validated["num_inference_steps"] = num_inference_steps
    
    # Validate guidance_scale (FLUX 2 Pro default is 4.5)
    if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0.1 or guidance_scale > 20.0:
        raise ValueError("guidance_scale must be a number between 0.1 and 20.0")
    validated["guidance_scale"] = float(guidance_scale)
    
    # Validate num_images
    if not isinstance(num_images, int) or num_images < 1 or num_images > 4:
        raise ValueError("num_images must be an integer between 1 and 4")
    validated["num_images"] = num_images
    
    # Validate output_format
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"Invalid output_format '{output_format}'. Must be one of: {VALID_OUTPUT_FORMATS}")
    validated["output_format"] = output_format
    
    # Validate acceleration
    if acceleration not in VALID_ACCELERATION_MODES:
        raise ValueError(f"Invalid acceleration '{acceleration}'. Must be one of: {VALID_ACCELERATION_MODES}")
    validated["acceleration"] = acceleration
    
    return validated


def _upscale_image(image_url: str, original_prompt: str) -> Dict[str, Any]:
    """
    Upscale an image using FAL.ai's Clarity Upscaler.
    
    Uses the synchronous fal_client API to avoid event loop lifecycle issues
    when called from threaded contexts (e.g. gateway thread pool).
    
    Args:
        image_url (str): URL of the image to upscale
        original_prompt (str): Original prompt used to generate the image
    
    Returns:
        Dict[str, Any]: Upscaled image data or None if upscaling fails
    """
    try:
        logger.info("Upscaling image with Clarity Upscaler...")
        
        # Prepare arguments for upscaler
        upscaler_arguments = {
            "image_url": image_url,
            "prompt": f"{UPSCALER_DEFAULT_PROMPT}, {original_prompt}",
            "upscale_factor": UPSCALER_FACTOR,
            "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
            "creativity": UPSCALER_CREATIVITY,
            "resemblance": UPSCALER_RESEMBLANCE,
            "guidance_scale": UPSCALER_GUIDANCE_SCALE,
            "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
            "enable_safety_checker": UPSCALER_SAFETY_CHECKER
        }
        
        # Use sync API — fal_client.submit() uses httpx.Client (no event loop).
        # The async API (submit_async) caches a global httpx.AsyncClient via
        # @cached_property, which breaks when asyncio.run() destroys the loop
        # between calls (gateway thread-pool pattern).
        handler = _submit_fal_request(
            UPSCALER_MODEL,
            arguments=upscaler_arguments,
        )
        
        # Get the upscaled result (sync — blocks until done)
        result = handler.get()
        
        if result and "image" in result:
            upscaled_image = result["image"]
            logger.info("Image upscaled successfully to %sx%s", upscaled_image.get('width', 'unknown'), upscaled_image.get('height', 'unknown'))
            return {
                "url": upscaled_image["url"],
                "width": upscaled_image.get("width", 0),
                "height": upscaled_image.get("height", 0),
                "upscaled": True,
                "upscale_factor": UPSCALER_FACTOR
            }
        else:
            logger.error("Upscaler returned invalid response")
            return None
            
    except Exception as e:
        logger.error("Error upscaling image: %s", e, exc_info=True)
        return None


def image_generate_tool(
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_images: int = DEFAULT_NUM_IMAGES,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    seed: Optional[int] = None,
    input_images: Optional[list[str]] = None,
) -> str:
    """
    Generate images from text prompts using the configured backend.

    Default behavior preserves the existing FAL.ai FLUX 2 Pro + auto-upscale path.
    When config.yaml sets image_generation.provider=openrouter, Hermes routes through
    OpenRouter image generation and materializes returned data URLs into local files.

    Args:
        prompt (str): The text prompt describing the desired image
        aspect_ratio (str): Image aspect ratio - "landscape", "square", or "portrait" (default: "landscape")
        num_inference_steps (int): Number of denoising steps (1-50, default: 50)
        guidance_scale (float): How closely to follow prompt (0.1-20.0, default: 4.5)
        num_images (int): Number of images to generate (1-4, default: 1)
        output_format (str): Image format "jpeg" or "png" (default: "png")
        seed (Optional[int]): Random seed for reproducible results (optional)
        input_images (Optional[list[str]]): Optional list of local file paths, URLs, or data URLs to use as image inputs for OpenRouter image-edit/compositing flows.

    Returns:
        str: JSON string containing minimal generation results:
             {
                 "success": bool,
                 "image": str or None  # URL/path of the generated image, or None if failed
             }
    """
    image_config = _load_image_generation_config()
    provider_error = str(image_config.get("provider_error") or "").strip()
    provider = _normalize_image_provider(image_config.get("provider")) if not provider_error else IMAGE_PROVIDER_FAL

    # Validate and map aspect_ratio to actual image_size
    aspect_ratio_lower = aspect_ratio.lower().strip() if aspect_ratio else DEFAULT_ASPECT_RATIO
    if aspect_ratio_lower not in ASPECT_RATIO_MAP:
        logger.warning("Invalid aspect_ratio '%s', defaulting to '%s'", aspect_ratio, DEFAULT_ASPECT_RATIO)
        aspect_ratio_lower = DEFAULT_ASPECT_RATIO
    image_size = ASPECT_RATIO_MAP[aspect_ratio_lower]

    debug_call_data = {
        "parameters": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "seed": seed,
            "input_images": list(input_images or []),
            "provider": provider,
            "provider_model": image_config.get("model"),
        },
        "error": None,
        "success": False,
        "images_generated": 0,
        "generation_time": 0
    }

    start_time = datetime.datetime.now()

    try:
        # Validate prompt
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")

        if provider_error:
            raise ValueError(provider_error)

        if provider == IMAGE_PROVIDER_OPENROUTER:
            return _generate_via_openrouter(
                prompt,
                aspect_ratio=aspect_ratio_lower,
                input_images=input_images,
                image_config=image_config,
                debug_call_data=debug_call_data,
                start_time=start_time,
            )

        if input_images:
            raise ValueError("input_images is currently supported only for the OpenRouter backend")

        logger.info("Generating %s image(s) with FLUX 2 Pro: %s", num_images, prompt[:80])

        # Check API key availability
        if not (os.getenv("FAL_KEY") or _resolve_managed_fal_gateway()):
            message = "FAL_KEY environment variable not set"
            if managed_nous_tools_enabled():
                message += " and managed FAL gateway is unavailable"
            raise ValueError(message)

        # Validate other parameters
        validated_params = _validate_parameters(
            image_size, num_inference_steps, guidance_scale, num_images, output_format, "none"
        )

        # Prepare arguments for FAL.ai FLUX 2 Pro API
        arguments = {
            "prompt": prompt.strip(),
            "image_size": validated_params["image_size"],
            "num_inference_steps": validated_params["num_inference_steps"],
            "guidance_scale": validated_params["guidance_scale"],
            "num_images": validated_params["num_images"],
            "output_format": validated_params["output_format"],
            "enable_safety_checker": ENABLE_SAFETY_CHECKER,
            "safety_tolerance": SAFETY_TOLERANCE,
            "sync_mode": True  # Use sync mode for immediate results
        }

        # Add seed if provided
        if seed is not None and isinstance(seed, int):
            arguments["seed"] = seed

        logger.info("Submitting generation request to FAL.ai FLUX 2 Pro...")
        logger.info("  Model: %s", DEFAULT_MODEL)
        logger.info("  Aspect Ratio: %s -> %s", aspect_ratio_lower, image_size)
        logger.info("  Steps: %s", validated_params['num_inference_steps'])
        logger.info("  Guidance: %s", validated_params['guidance_scale'])

        # Submit request to FAL.ai using sync API (avoids cached event loop issues)
        handler = _submit_fal_request(
            DEFAULT_MODEL,
            arguments=arguments,
        )

        # Get the result (sync — blocks until done)
        result = handler.get()

        generation_time = (datetime.datetime.now() - start_time).total_seconds()

        # Process the response
        if not result or "images" not in result:
            raise ValueError("Invalid response from FAL.ai API - no images returned")

        images = result.get("images", [])
        if not images:
            raise ValueError("No images were generated")

        # Format image data and upscale images
        formatted_images = []
        for img in images:
            if isinstance(img, dict) and "url" in img:
                original_image = {
                    "url": img["url"],
                    "width": img.get("width", 0),
                    "height": img.get("height", 0)
                }

                # Attempt to upscale the image
                upscaled_image = _upscale_image(img["url"], prompt.strip())

                if upscaled_image:
                    # Use upscaled image if successful
                    formatted_images.append(upscaled_image)
                else:
                    # Fall back to original image if upscaling fails
                    logger.warning("Using original image as fallback")
                    original_image["upscaled"] = False
                    formatted_images.append(original_image)

        if not formatted_images:
            raise ValueError("No valid image URLs returned from API")

        upscaled_count = sum(1 for img in formatted_images if img.get("upscaled", False))
        logger.info("Generated %s image(s) in %.1fs (%s upscaled)", len(formatted_images), generation_time, upscaled_count)

        # Prepare successful response - minimal format
        response_data = {
            "success": True,
            "image": formatted_images[0]["url"] if formatted_images else None
        }

        debug_call_data["success"] = True
        debug_call_data["images_generated"] = len(formatted_images)
        debug_call_data["generation_time"] = generation_time

        # Log debug information
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = f"Error generating image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)

        # Include error details so callers can diagnose failures
        response_data = {
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }

        debug_call_data["error"] = error_msg
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()

        return json.dumps(response_data, indent=2, ensure_ascii=False)


def check_fal_api_key() -> bool:
    """
    Check if the FAL.ai API key is available in environment variables.

    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("FAL_KEY") or _resolve_managed_fal_gateway())


def check_image_generation_requirements() -> bool:
    """
    Check if all requirements for image generation tools are met.

    Returns:
        bool: True if requirements are met, False otherwise
    """
    image_config = _load_image_generation_config()
    provider_error = str(image_config.get("provider_error") or "").strip()
    if provider_error:
        return False

    provider = _normalize_image_provider(image_config.get("provider"))

    if provider == IMAGE_PROVIDER_OPENROUTER:
        return _openrouter_api_key_available(image_config)

    try:
        # Check API key
        if not check_fal_api_key():
            return False

        # Check if fal_client is available
        import fal_client  # noqa: F401 — SDK presence check
        return True

    except ImportError:
        return False



if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🎨 Image Generation Tools Module - FLUX 2 Pro + Auto Upscaling")
    print("=" * 60)
    
    # Check if API key is available
    api_available = check_fal_api_key()
    
    if not api_available:
        print("❌ FAL_KEY environment variable not set")
        print("Please set your API key: export FAL_KEY='your-key-here'")
        print("Get API key at: https://fal.ai/")
        exit(1)
    else:
        print("✅ FAL.ai API key found")
    
    # Check if fal_client is available
    try:
        import fal_client
        print("✅ fal_client library available")
    except ImportError:
        print("❌ fal_client library not found")
        print("Please install: pip install fal-client")
        exit(1)
    
    print("🛠️ Image generation tools ready for use!")
    print(f"🤖 Using model: {DEFAULT_MODEL}")
    print(f"🔍 Auto-upscaling with: {UPSCALER_MODEL} ({UPSCALER_FACTOR}x)")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/image_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set IMAGE_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from image_generation_tool import image_generate_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      # Generate image with automatic 2x upscaling")
    print("      result = await image_generate_tool(")
    print("          prompt='A serene mountain landscape with cherry blossoms',")
    print("          image_size='landscape_4_3',")
    print("          num_images=1")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nSupported image sizes:")
    for size in VALID_IMAGE_SIZES:
        print(f"  - {size}")
    print("  - Custom: {'width': 512, 'height': 768} (if needed)")
    
    print("\nAcceleration modes:")
    for mode in VALID_ACCELERATION_MODES:
        print(f"  - {mode}")
    
    print("\nExample prompts:")
    print("  - 'A candid street photo of a woman with a pink bob and bold eyeliner'")
    print("  - 'Modern architecture building with glass facade, sunset lighting'")
    print("  - 'Abstract art with vibrant colors and geometric patterns'")
    print("  - 'Portrait of a wise old owl perched on ancient tree branch'")
    print("  - 'Futuristic cityscape with flying cars and neon lights'")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export IMAGE_TOOLS_DEBUG=true")
    print("  # Debug logs capture all image generation calls and results")
    print("  # Logs saved to: ./logs/image_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate",
    "description": "Generate high-quality images from text prompts using the configured backend. FAL preserves the legacy FLUX+upscale path; OpenRouter routes through supported image models and returns a local cached file when providers emit base64 data URLs.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the desired image. Be detailed and descriptive."
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "square", "portrait"],
                "description": "Requested aspect ratio. Hermes maps this to the backend's native size/aspect controls when supported.",
                "default": "landscape"
            },
            "num_inference_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Number of inference steps. Supported by FAL backend; currently ignored by OpenRouter image backends.",
                "default": 50
            },
            "guidance_scale": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 20.0,
                "description": "Prompt guidance scale. Supported by FAL backend; currently ignored by OpenRouter image backends.",
                "default": 4.5
            },
            "num_images": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
                "description": "Number of images to generate. Currently Hermes returns the first successful image in the response.",
                "default": 1
            },
            "output_format": {
                "type": "string",
                "enum": ["jpeg", "png"],
                "description": "Preferred output format. Supported by FAL backend; OpenRouter returns the provider-native encoded image.",
                "default": "png"
            },
            "seed": {
                "type": ["integer", "null"],
                "description": "Optional random seed for reproducible generations when supported by the backend.",
                "default": None
            },
            "input_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional image inputs for image-to-image or compositing flows. Each entry may be an absolute local file path, an http/https URL, or a data:image/... URL. Currently supported only on the OpenRouter backend.",
                "default": []
            }
        },
        "required": ["prompt"]
    }
}


def _handle_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for image generation")
    return image_generate_tool(
        prompt=prompt,
        aspect_ratio=args.get("aspect_ratio", "landscape"),
        num_inference_steps=args.get("num_inference_steps", 50),
        guidance_scale=args.get("guidance_scale", 4.5),
        num_images=args.get("num_images", 1),
        output_format=args.get("output_format", "png"),
        seed=args.get("seed"),
        input_images=args.get("input_images"),
    )


registry.register(
    name="image_generate",
    toolset="image_gen",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate,
    check_fn=check_image_generation_requirements,
    requires_env=[],
    is_async=False,  # Switched to sync fal_client API to fix "Event loop is closed" in gateway
    emoji="🎨",
)
