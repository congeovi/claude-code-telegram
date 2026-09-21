"""Dynamic Claude model catalog fetched directly from Claude API / OAuth session."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import structlog

logger = structlog.get_logger()

# In-memory cache for models
_CACHED_MODELS: List[Dict[str, Any]] = []
_LAST_FETCH_TIME: float = 0
_CACHE_TTL_SECONDS: float = 600.0  # 10 minutes


def _get_auth_headers(api_key: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Retrieve auth headers from API key or Claude Code OAuth session."""
    headers = {"anthropic-version": "2023-06-01"}

    # 1. Check direct API key argument or environment
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key and key.strip():
        headers["x-api-key"] = key.strip()
        return headers

    # 2. Check ~/.claude/.credentials.json (Standard Claude Code OAuth token)
    cred_path = Path.home() / ".claude" / ".credentials.json"
    if cred_path.is_file():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            token = data.get("claudeAiOauth", {}).get("accessToken")
            if token and token.strip():
                headers["Authorization"] = f"Bearer {token.strip()}"
                return headers
        except Exception as e:
            logger.debug("Failed reading .credentials.json", error=str(e))

    # 3. Check ~/.claude/settings.json
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.is_file():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            token = data.get("env", {}).get("CLAUDE_CODE_OAUTH_TOKEN")
            if token and token.strip():
                headers["Authorization"] = f"Bearer {token.strip()}"
                return headers
        except Exception as e:
            logger.debug("Failed reading settings.json", error=str(e))

    return None


def _format_model_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Format and enrich raw model entry with display metadata."""
    model_id = raw.get("id", "")
    display_name = raw.get("display_name", model_id)
    created_at = raw.get("created_at", "")
    m_lower = model_id.lower()

    if "sonnet" in m_lower:
        icon = "🟣"
        family = "sonnet"
    elif "opus" in m_lower:
        icon = "🧠"
        family = "opus"
    elif "haiku" in m_lower:
        icon = "⚡"
        family = "haiku"
    elif "fable" in m_lower or "mythos" in m_lower:
        icon = "✨"
        family = "fable"
    else:
        icon = "🤖"
        family = "claude"

    return {
        "id": model_id,
        "name": display_name,
        "icon": icon,
        "family": family,
        "created_at": created_at,
        "max_tokens": raw.get("max_tokens", 128000),
    }


def get_cached_models() -> List[Dict[str, Any]]:
    """Return currently cached models or default fallback if empty."""
    global _CACHED_MODELS
    if _CACHED_MODELS:
        return list(_CACHED_MODELS)
    # Default fallback if not yet fetched
    return [
        {"id": "claude-sonnet-5", "name": "Claude Sonnet 5", "icon": "🟣", "family": "sonnet"},
        {"id": "claude-opus-5", "name": "Claude Opus 5", "icon": "🧠", "family": "opus"},
        {"id": "claude-fable-5-1", "name": "Claude Fable 5.1", "icon": "✨", "family": "fable"},
        {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "icon": "⚡", "family": "haiku"},
    ]


async def fetch_claude_models(
    api_key: Optional[str] = None, force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """Fetch real-time available models directly from Claude API."""
    global _CACHED_MODELS, _LAST_FETCH_TIME

    now = time.time()
    if not force_refresh and _CACHED_MODELS and (now - _LAST_FETCH_TIME < _CACHE_TTL_SECONDS):
        return list(_CACHED_MODELS)

    headers = _get_auth_headers(api_key)
    if not headers:
        logger.warning("No Claude auth token found for fetching models, using cache")
        return get_cached_models()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

        if data:
            # Sort newest first by created_at
            data.sort(key=lambda m: m.get("created_at", ""), reverse=True)
            formatted = [_format_model_entry(m) for m in data if m.get("id")]
            if formatted:
                _CACHED_MODELS = formatted
                _LAST_FETCH_TIME = now
                logger.info(
                    "Fetched latest models directly from Claude",
                    count=len(formatted),
                    latest_model=formatted[0]["id"],
                )
                return list(_CACHED_MODELS)

    except Exception as e:
        logger.warning("Failed fetching models from Claude API", error=str(e))

    return get_cached_models()


def resolve_model_id(target: str, models: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Resolve alias or user input (e.g. 'sonnet', 'opus', '5') to exact latest model ID."""
    target_clean = target.strip().lower()
    if target_clean in ("default", "clear", "reset"):
        return None

    model_list = models or get_cached_models()

    # Exact ID match
    for m in model_list:
        if m["id"].lower() == target_clean:
            return m["id"]

    # Family match (since list is sorted newest first, first match is latest version!)
    if target_clean in ("sonnet", "latest-sonnet"):
        for m in model_list:
            if m.get("family") == "sonnet":
                return m["id"]

    if target_clean in ("opus", "latest-opus"):
        for m in model_list:
            if m.get("family") == "opus":
                return m["id"]

    if target_clean in ("haiku", "latest-haiku"):
        for m in model_list:
            if m.get("family") == "haiku":
                return m["id"]

    if target_clean in ("fable", "latest-fable"):
        for m in model_list:
            if m.get("family") == "fable":
                return m["id"]

    # Fuzzy substring in name or id
    for m in model_list:
        if target_clean in m["id"].lower() or target_clean in m["name"].lower():
            return m["id"]

    # Fallback to whatever user passed
    return target.strip()
