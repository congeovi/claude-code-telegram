"""Tests for Antigravity-style CLI features in Claude Code Telegram bot."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Update
from telegram.ext import ContextTypes

from src.bot.keyboards import (
    SUPPORTED_MODELS,
    SUPPORTED_MODES,
    menu_keyboard,
    mode_keyboard,
    model_keyboard,
    tokens_keyboard,
    workspace_keyboard,
)
from src.bot.orchestrator import MessageOrchestrator
from src.config import create_test_config


@pytest.fixture
def agentic_settings(tmp_path):
    return create_test_config(approved_directory=str(tmp_path), agentic_mode=True)


@pytest.fixture
def deps():
    return {
        "claude_integration": MagicMock(),
        "storage": MagicMock(),
        "security_validator": MagicMock(),
        "rate_limiter": MagicMock(),
        "audit_logger": MagicMock(),
    }


def test_keyboards_structure():
    """Verify inline keyboard structures and contents."""
    # Model keyboard
    first_model = SUPPORTED_MODELS[0]["id"]
    first_name = SUPPORTED_MODELS[0]["name"]
    kb_model = model_keyboard(first_model)
    buttons_text = [b[0].text for b in kb_model.inline_keyboard]
    assert any(first_name in t and "✅" in t for t in buttons_text)

    # Mode keyboard
    kb_mode = mode_keyboard("plan")
    assert len(kb_mode.inline_keyboard) == len(SUPPORTED_MODES) + 1
    mode_buttons = [b[0].text for b in kb_mode.inline_keyboard]
    assert any("Plan" in t and "✅" in t for t in mode_buttons)

    # Menu keyboard
    kb_menu = menu_keyboard()
    assert len(kb_menu.inline_keyboard) == 4
    all_callbacks = [
        btn.callback_data for row in kb_menu.inline_keyboard for btn in row
    ]
    assert "menu:model" in all_callbacks
    assert "menu:mode" in all_callbacks
    assert "menu:workspace" in all_callbacks
    assert "action:tokens" in all_callbacks
    assert "action:compact" in all_callbacks
    assert "action:cancel" in all_callbacks

    # Tokens keyboard
    kb_tokens = tokens_keyboard()
    assert len(kb_tokens.inline_keyboard) == 2


def test_workspace_keyboard(tmp_path):
    """Test workspace keyboard directories list and pagination."""
    (tmp_path / "proj1").mkdir()
    (tmp_path / "proj2").mkdir()
    (tmp_path / "proj3").mkdir()

    kb = workspace_keyboard(tmp_path / "proj1", tmp_path, 0)
    flat_buttons = [btn.text for row in kb.inline_keyboard for btn in row]
    assert any("proj1" in b for b in flat_buttons)
    assert any("proj2" in b for b in flat_buttons)


@pytest.mark.asyncio
async def test_agentic_model_command_and_callback(agentic_settings, deps):
    """Test /model command and set:model: callback."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_message.text = "/model sonnet"
    update.effective_message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {}

    # Run /model sonnet
    await orchestrator.agentic_model(update, context)
    assert "sonnet" in context.user_data["claude_model"].lower()
    assert update.effective_message.reply_text.called

    # Callback selection
    query_update = MagicMock()
    query_update.callback_query.data = "set:model:claude-3-5-haiku-20241022"
    query_update.callback_query.answer = AsyncMock()
    query_update.callback_query.edit_message_text = AsyncMock()

    await orchestrator._handle_set_callback(query_update, context)
    assert context.user_data["claude_model"] == "claude-3-5-haiku-20241022"
    assert query_update.callback_query.answer.called


@pytest.mark.asyncio
async def test_agentic_mode_command_and_callback(agentic_settings, deps):
    """Test /mode command and set:mode: callback."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_message.text = "/mode plan"
    update.effective_message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {}

    # Run /mode plan
    await orchestrator.agentic_mode(update, context)
    assert context.user_data["claude_mode"] == "plan"

    # Callback switch to acceptEdits
    query_update = MagicMock()
    query_update.callback_query.data = "set:mode:acceptEdits"
    query_update.callback_query.answer = AsyncMock()
    query_update.callback_query.edit_message_text = AsyncMock()

    await orchestrator._handle_set_callback(query_update, context)
    assert context.user_data["claude_mode"] == "acceptEdits"


@pytest.mark.asyncio
async def test_agentic_stop_command(agentic_settings, deps):
    """Test /stop and /cancel interrupt active requests."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    user_id = 999
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_message.reply_text = AsyncMock()

    context = MagicMock()

    # No active request
    await orchestrator.agentic_stop(update, context)
    assert "Hiện không có tác vụ nào" in update.effective_message.reply_text.call_args[0][0]

    # With active request
    import asyncio
    from src.bot.orchestrator import ActiveRequest
    event = asyncio.Event()
    progress_msg = MagicMock()
    progress_msg.edit_text = AsyncMock()
    orchestrator._active_requests[user_id] = ActiveRequest(
        user_id=user_id,
        interrupt_event=event,
        progress_msg=progress_msg,
    )

    await orchestrator.agentic_stop(update, context)
    assert event.is_set()
    assert "ngắt tác vụ" in update.effective_message.reply_text.call_args[0][0]


@pytest.mark.asyncio
async def test_agentic_tokens_reporting(agentic_settings, deps):
    """Test /tokens reporting with usage statistics."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {
        "claude_session_id": "test-session-1234567890",
        "session_usage": {
            "input_tokens": 1500,
            "output_tokens": 500,
            "cache_read_tokens": 200,
            "cache_creation_tokens": 100,
            "total_cost": 0.015,
            "turns": 2,
        },
    }

    await orchestrator.agentic_tokens(update, context)
    called_text = update.effective_message.reply_text.call_args[0][0]
    assert "Báo Cáo Token & Chi Phí" in called_text
    assert "1,500" in called_text
    assert "500" in called_text
    assert "$0.0150" in called_text


@pytest.mark.asyncio
async def test_agentic_workspace_command(agentic_settings, deps, tmp_path):
    """Test /workspace command and directory switching."""
    agentic_settings.approved_directory = tmp_path
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    sub = tmp_path / "sub_project"
    sub.mkdir()

    update = MagicMock()
    update.effective_user.id = 111
    update.effective_message.text = f"/workspace {sub.name}"
    update.effective_message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {}
    context.bot_data = {}

    await orchestrator.agentic_workspace(update, context)
    assert context.user_data["current_directory"] == sub

    # Reset
    update.effective_message.text = "/workspace reset"
    await orchestrator.agentic_workspace(update, context)
    assert context.user_data["current_directory"] == tmp_path


@pytest.mark.asyncio
async def test_agentic_menu_command(agentic_settings, deps):
    """Test /menu and /help commands return master keyboard."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_message.reply_text = AsyncMock()
    context = MagicMock()

    await orchestrator.agentic_menu(update, context)
    assert update.effective_message.reply_text.called
    assert update.effective_message.reply_text.call_args[1]["reply_markup"] is not None

    await orchestrator.agentic_help(update, context)
    assert update.effective_message.reply_text.called


@pytest.mark.asyncio
async def test_agentic_compact_callback_query(agentic_settings, deps):
    """Test /compact when called from callback query sends clean /compact prompt."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    mock_resp = MagicMock()
    mock_resp.content = "Compacted"
    mock_resp.session_id = "s1"
    deps["claude_integration"].run_command = AsyncMock(return_value=mock_resp)

    update = MagicMock()
    update.effective_user.id = 123
    update.callback_query = MagicMock()
    update.callback_query.data = "action:compact"
    update.effective_message.text = "📊 Báo Cáo Token & Chi Phí Phiên Làm Việc\n\n🆔 Session..."
    progress_msg = MagicMock()
    progress_msg.edit_text = AsyncMock()
    update.effective_message.reply_text = AsyncMock(return_value=progress_msg)

    context = MagicMock()
    context.user_data = {"claude_session_id": "test-session"}
    context.bot_data = deps

    await orchestrator.agentic_compact(update, context)

    # Must send "/compact" prompt, not "/compact Báo Cáo Token..."
    called_prompt = deps["claude_integration"].run_command.call_args[1]["prompt"]
    assert called_prompt == "/compact"


@pytest.mark.asyncio
async def test_agentic_mode_supports_underscore(agentic_settings, deps):
    """Test /mode accept_edits supports underscore."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_message.text = "/mode accept_edits"
    update.effective_message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {}

    await orchestrator.agentic_mode(update, context)
    assert context.user_data["claude_mode"] == "acceptEdits"

