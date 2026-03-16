# -*- coding: utf-8 -*-
"""
OpenAI Responses API adapter for Kiro Gateway.

Translates between the OpenAI Responses API wire format (POST /v1/responses)
and the existing Chat Completions pipeline. This enables clients like
OpenAI Codex CLI that only speak the Responses API protocol.
"""

import json
import uuid
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from kiro.routes_openai import verify_api_key

router = APIRouter()


# ---------------------------------------------------------------------------
# Request conversion: Responses API -> Chat Completions
# ---------------------------------------------------------------------------

def _content_items_to_text(content_items: Any) -> str:
    """Extract plain text from a Responses API content array or string."""
    if isinstance(content_items, str):
        return content_items
    if isinstance(content_items, list):
        parts = []
        for item in content_items:
            if isinstance(item, dict):
                if item.get("type") in ("input_text", "output_text"):
                    parts.append(item.get("text", ""))
                elif item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content_items) if content_items else ""


def _responses_input_to_messages(input_items: Any, instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert Responses API ``input`` to Chat Completions ``messages``.

    Handles:
    - Plain string input (becomes a single user message)
    - List of ResponseItem objects (message, function_call, function_call_output, etc.)
    """
    messages: List[Dict[str, Any]] = []

    # System message from instructions
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Simple string input
    if isinstance(input_items, str):
        messages.append({"role": "user", "content": input_items})
        return messages

    if not isinstance(input_items, list):
        messages.append({"role": "user", "content": str(input_items)})
        return messages

    # Walk through ResponseItem list
    pending_tool_calls: List[Dict[str, Any]] = []

    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")

        if item_type == "message":
            # Flush any pending tool calls as an assistant message first
            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []

            role = item.get("role", "user")
            text = _content_items_to_text(item.get("content", ""))

            # Map Responses API roles to Chat Completions roles
            if role == "system":
                messages.append({"role": "system", "content": text})
            elif role == "developer":
                messages.append({"role": "system", "content": text})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": text})
            else:
                messages.append({"role": "user", "content": text})

        elif item_type == "function_call":
            pending_tool_calls.append({
                "id": item.get("call_id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })

        elif item_type == "function_call_output":
            # Flush pending tool calls first
            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []

            output = item.get("output", "")
            if isinstance(output, dict):
                body = output.get("body", "")
                if isinstance(body, list):
                    # ContentItems list
                    output = _content_items_to_text(body)
                else:
                    output = str(body) if body else ""

            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": output if isinstance(output, str) else json.dumps(output),
            })

        elif item_type == "reasoning":
            # Skip reasoning items in input — they're informational
            pass

        elif item_type == "custom_tool_call":
            pending_tool_calls.append({
                "id": item.get("call_id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("input", "{}"),
                },
            })

        elif item_type == "custom_tool_call_output":
            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []

            output = item.get("output", "")
            if isinstance(output, dict):
                body = output.get("body", "")
                output = str(body) if body else ""

            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": output if isinstance(output, str) else json.dumps(output),
            })

    # Flush remaining tool calls
    if pending_tool_calls:
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": pending_tool_calls,
        })

    return messages


def _convert_tools(tools: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Normalize Responses API tools to standard OpenAI Chat Completions format.

    Codex sends tools in flat format:
      {"type": "function", "name": "foo", "parameters": {...}, "description": "..."}

    The gateway expects standard OpenAI format:
      {"type": "function", "function": {"name": "foo", "parameters": {...}, "description": "..."}}
    """
    if not tools:
        return None
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized.append(tool)
            continue
        # Already in standard format with nested "function" key
        if "function" in tool and isinstance(tool["function"], dict):
            normalized.append(tool)
            continue
        # Flat format: wrap name/description/parameters into "function"
        if "name" in tool:
            func = {
                "name": tool["name"],
            }
            if "description" in tool:
                func["description"] = tool["description"]
            if "parameters" in tool:
                func["parameters"] = tool["parameters"]
            elif "input_schema" in tool:
                func["parameters"] = tool["input_schema"]
            normalized.append({
                "type": tool.get("type", "function"),
                "function": func,
            })
        else:
            normalized.append(tool)
    return normalized


# ---------------------------------------------------------------------------
# Response conversion: Chat Completions SSE -> Responses API SSE
# ---------------------------------------------------------------------------

def _make_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


async def _stream_as_responses_api(
    chat_stream,
    response_id: str,
    model: str,
):
    """
    Wrap a Chat Completions SSE stream and re-emit as Responses API SSE events.
    """
    created = int(time.time())

    # Emit response.created
    yield _sse("response.created", {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "model": model,
            "status": "in_progress",
            "output": [],
        },
    })

    # Track state
    output_items: List[Dict[str, Any]] = []
    full_text = ""
    tool_calls_map: Dict[int, Dict[str, Any]] = {}  # index -> tool_call
    usage_data = None
    message_item_added = False
    text_content_added = False

    async for chunk_str in chat_stream:
        if not chunk_str.startswith("data:"):
            continue
        data_str = chunk_str[len("data:"):].strip()
        if not data_str or data_str == "[DONE]":
            if data_str == "[DONE]":
                # Will emit completed below
                pass
            continue

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            # Could be a usage-only chunk
            if "usage" in chunk:
                usage_data = chunk["usage"]
            continue

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Text content delta
        text_delta = delta.get("content")
        if text_delta:
            if not message_item_added:
                message_item_added = True
                yield _sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                })
            if not text_content_added:
                text_content_added = True
                yield _sse("response.content_part.added", {
                    "type": "response.content_part.added",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                })

            full_text += text_delta
            yield _sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": text_delta,
            })

        # Tool calls delta
        tc_deltas = delta.get("tool_calls")
        if tc_deltas:
            for tc in tc_deltas:
                idx = tc.get("index", 0)
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "name": "",
                        "arguments": "",
                    }
                entry = tool_calls_map[idx]
                if tc.get("id"):
                    entry["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    entry["name"] += func["name"]
                if func.get("arguments"):
                    entry["arguments"] += func["arguments"]

        # Usage in final chunk
        if "usage" in chunk:
            usage_data = chunk["usage"]

        # Finish
        if finish_reason:
            # Close text content if we had any
            if text_content_added:
                yield _sse("response.output_text.done", {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "text": full_text,
                })
                yield _sse("response.content_part.done", {
                    "type": "response.content_part.done",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": full_text},
                })

            if message_item_added:
                msg_item = {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full_text}] if full_text else [],
                }
                output_items.append(msg_item)
                yield _sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": msg_item,
                })

            # Emit function_call items
            for idx in sorted(tool_calls_map.keys()):
                tc = tool_calls_map[idx]
                out_idx = len(output_items)
                fc_item = {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "status": "completed",
                }
                output_items.append(fc_item)
                yield _sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": out_idx,
                    "item": fc_item,
                })
                yield _sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": out_idx,
                    "item": fc_item,
                })

    # Build usage
    resp_usage = None
    if usage_data:
        resp_usage = {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }

    # Emit response.completed
    yield _sse("response.completed", {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "model": model,
            "status": "completed",
            "output": output_items,
            "usage": resp_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
    })


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Non-streaming response conversion
# ---------------------------------------------------------------------------

def _chat_completion_to_response(chat_resp: dict, response_id: str) -> dict:
    """Convert a Chat Completions JSON response to Responses API format."""
    output_items = []
    choices = chat_resp.get("choices", [])
    if choices:
        choice = choices[0]
        msg = choice.get("message", {})

        # Text content
        text = msg.get("content", "")
        if text:
            output_items.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            })

        # Tool calls
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            output_items.append({
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": func.get("arguments", "{}"),
                "status": "completed",
            })

    usage = chat_resp.get("usage", {})
    return {
        "id": response_id,
        "object": "response",
        "created_at": chat_resp.get("created", int(time.time())),
        "model": chat_resp.get("model", ""),
        "status": "completed",
        "output": output_items,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def responses_api(request: Request):
    """
    OpenAI Responses API endpoint.

    Accepts Responses API requests and proxies them through the existing
    Chat Completions pipeline.
    """
    body = await request.json()
    logger.info(f"Request to /v1/responses (model={body.get('model')}, stream={body.get('stream', True)})")

    model = body.get("model", "")
    instructions = body.get("instructions", "")
    input_items = body.get("input", [])
    tools = body.get("tools")
    stream = body.get("stream", True)  # Responses API defaults to streaming
    temperature = body.get("temperature")
    max_tokens = body.get("max_output_tokens") or body.get("max_tokens")
    tool_choice = body.get("tool_choice", "auto")

    # Convert to Chat Completions format
    messages = _responses_input_to_messages(input_items, instructions)
    if not messages:
        raise HTTPException(status_code=400, detail="No input messages")

    chat_request_body = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if tools:
        chat_request_body["tools"] = _convert_tools(tools)
        chat_request_body["tool_choice"] = tool_choice
    if temperature is not None:
        chat_request_body["temperature"] = temperature
    if max_tokens is not None:
        chat_request_body["max_tokens"] = max_tokens

    logger.debug(f"Converted to chat completions: {len(messages)} messages, stream={stream}")

    # --- Call the existing Chat Completions pipeline internally ---
    from kiro.models_openai import ChatCompletionRequest, ChatMessage
    from kiro.converters_openai import build_kiro_payload
    from kiro.streaming_openai import stream_kiro_to_openai, collect_stream_response
    from kiro.http_client import KiroHttpClient
    from kiro.utils import generate_conversation_id
    from kiro.auth import KiroAuthManager, AuthType
    from kiro.cache import ModelInfoCache
    from kiro.config import HIDDEN_MODELS

    auth_manager: KiroAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache

    # Build ChatCompletionRequest from our converted data
    chat_req = ChatCompletionRequest(**chat_request_body)

    conversation_id = generate_conversation_id()

    profile_arn_for_payload = ""
    if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
        profile_arn_for_payload = auth_manager.profile_arn

    try:
        kiro_payload = build_kiro_payload(chat_req, conversation_id, profile_arn_for_payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    url = f"{auth_manager.api_host}/generateAssistantResponse"
    response_id = _make_response_id()

    if stream:
        http_client = KiroHttpClient(auth_manager, shared_client=None)
    else:
        shared_client = request.app.state.http_client
        http_client = KiroHttpClient(auth_manager, shared_client=shared_client)

    try:
        kiro_response = await http_client.request_with_retry("POST", url, kiro_payload, stream=True)

        if kiro_response.status_code != 200:
            try:
                error_content = await kiro_response.aread()
            except Exception:
                error_content = b"Unknown error"
            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")
            logger.warning(f"Kiro API error: {kiro_response.status_code} - {error_text[:200]}")
            return JSONResponse(
                status_code=kiro_response.status_code,
                content={
                    "error": {
                        "message": error_text,
                        "type": "kiro_api_error",
                        "code": kiro_response.status_code,
                    }
                },
            )

        messages_for_tokenizer = [m if isinstance(m, dict) else m.model_dump() for m in chat_req.messages]
        tools_for_tokenizer = [t.model_dump() for t in chat_req.tools] if chat_req.tools else None

        if stream:
            # Wrap the chat completions stream into Responses API SSE
            chat_stream = stream_kiro_to_openai(
                http_client.client,
                kiro_response,
                model,
                model_cache,
                auth_manager,
                request_messages=messages_for_tokenizer,
                request_tools=tools_for_tokenizer,
            )

            async def response_stream_wrapper():
                try:
                    async for chunk in _stream_as_responses_api(chat_stream, response_id, model):
                        yield chunk
                except Exception as e:
                    logger.error(f"Responses API streaming error: {e}", exc_info=True)
                    raise
                finally:
                    await http_client.close()

            return StreamingResponse(response_stream_wrapper(), media_type="text/event-stream")

        else:
            chat_resp = await collect_stream_response(
                http_client.client,
                kiro_response,
                model,
                model_cache,
                auth_manager,
                request_messages=messages_for_tokenizer,
                request_tools=tools_for_tokenizer,
            )
            await http_client.close()
            return JSONResponse(content=_chat_completion_to_response(chat_resp, response_id))

    except HTTPException:
        await http_client.close()
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error in /v1/responses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
