"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import structlog
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..claude.models import fetch_claude_models, get_cached_models, resolve_model_id
from ..claude.sdk_integration import StreamUpdate
from ..config.settings import Settings
from ..projects import PrivateTopicsUnavailableError
from .keyboards import (
    SUPPORTED_MODELS,
    SUPPORTED_MODES,
    menu_keyboard,
    mode_keyboard,
    model_keyboard,
    tokens_keyboard,
    workspace_keyboard,
)
from .utils.draft_streamer import DraftStreamer, generate_draft_id
from .utils.html_format import escape_html
from .utils.image_extractor import (
    ImageAttachment,
    should_send_as_photo,
    validate_image_path,
)

logger = structlog.get_logger()

_MEDIA_TYPE_MAP = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _format_relative_time(dt: datetime) -> str:
    """Human-friendly 'time ago' label for a session timestamp."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"

# Patterns that look like secrets/credentials in CLI arguments
_SECRET_PATTERNS: List[re.Pattern[str]] = [
    # API keys / tokens (sk-ant-..., sk-..., ghp_..., gho_..., github_pat_..., xoxb-...)
    re.compile(
        r"(sk-ant-api\d*-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*"
        r"|(sk-[A-Za-z0-9_-]{20})[A-Za-z0-9_-]*"
        r"|(ghp_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(gho_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(github_pat_[A-Za-z0-9_]{5})[A-Za-z0-9_]*"
        r"|(xoxb-[A-Za-z0-9]{5})[A-Za-z0-9-]*"
    ),
    # AWS access keys
    re.compile(r"(AKIA[0-9A-Z]{4})[0-9A-Z]{12}"),
    # Generic long hex/base64 tokens after common flags/env patterns
    re.compile(
        r"((?:--token|--secret|--password|--api-key|--apikey|--auth)"
        r"[= ]+)['\"]?[A-Za-z0-9+/_.:-]{8,}['\"]?"
    ),
    # Inline env assignments like KEY=value
    re.compile(
        r"((?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|AUTH_TOKEN|PRIVATE_KEY"
        r"|ACCESS_KEY|CLIENT_SECRET|WEBHOOK_SECRET)"
        r"=)['\"]?[^\s'\"]{8,}['\"]?"
    ),
    # Bearer / Basic auth headers
    re.compile(r"(Bearer )[A-Za-z0-9+/_.:-]{8,}" r"|(Basic )[A-Za-z0-9+/=]{8,}"),
    # Connection strings with credentials  user:pass@host
    re.compile(r"://([^:]+:)[^@]{4,}(@)"),
]


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda m: next((g + "***" for g in m.groups() if g is not None), "***"),
            result,
        )
    return result


# Tool name -> friendly emoji mapping for verbose output
_TOOL_ICONS: Dict[str, str] = {
    "Read": "\U0001f4d6",
    "Write": "\u270f\ufe0f",
    "Edit": "\u270f\ufe0f",
    "MultiEdit": "\u270f\ufe0f",
    "Bash": "\U0001f4bb",
    "Glob": "\U0001f50d",
    "Grep": "\U0001f50d",
    "LS": "\U0001f4c2",
    "Task": "\U0001f9e0",
    "TaskOutput": "\U0001f9e0",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f310",
    "NotebookRead": "\U0001f4d3",
    "NotebookEdit": "\U0001f4d3",
    "TodoRead": "\u2611\ufe0f",
    "TodoWrite": "\u2611\ufe0f",
}


def _tool_icon(name: str) -> str:
    """Return emoji for a tool, with a default wrench."""
    return _TOOL_ICONS.get(name, "\U0001f527")


@dataclass
class ActiveRequest:
    """Tracks an in-flight Claude request so it can be interrupted."""

    user_id: int
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    interrupted: bool = False
    progress_msg: Any = None  # telegram Message object


@dataclass
class PendingToolApproval:
    """Tracks an in-flight interactive tool-approval prompt."""

    user_id: int
    future: "asyncio.Future[bool]"


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps
        self._active_requests: Dict[int, ActiveRequest] = {}
        self._pending_tool_approvals: Dict[str, PendingToolApproval] = {}
        self._known_commands: frozenset[str] = frozenset()

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            context.user_data.pop("_thread_context", None)

            is_sync_bypass = handler.__name__ == "sync_threads"
            is_start_bypass = handler.__name__ in {"start_command", "agentic_start"}
            message_thread_id = self._extract_message_thread_id(update)
            should_enforce = self.settings.enable_project_threads

            if should_enforce:
                if self.settings.project_threads_mode == "private":
                    should_enforce = not is_sync_bypass and not (
                        is_start_bypass and message_thread_id is None
                    )
                else:
                    should_enforce = not is_sync_bypass

            if should_enforce:
                allowed = await self._apply_thread_routing_context(update, context)
                if not allowed:
                    return

            try:
                await handler(update, context)
            finally:
                if should_enforce:
                    self._persist_thread_state(context)

        return wrapped

    async def _apply_thread_routing_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Enforce strict project-thread routing and load thread-local state."""
        manager = context.bot_data.get("project_threads_manager")
        if manager is None:
            await self._reject_for_thread_mode(
                update,
                "❌ <b>Project Thread Mode Misconfigured</b>\n\n"
                "Thread manager is not initialized.",
            )
            return False

        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return False

        if self.settings.project_threads_mode == "group":
            if chat.id != self.settings.project_threads_chat_id:
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False
        else:
            if getattr(chat, "type", "") != "private":
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False

        message_thread_id = self._extract_message_thread_id(update)
        if not message_thread_id:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        project = await manager.resolve_project(chat.id, message_thread_id)
        if not project:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        state_key = f"{chat.id}:{message_thread_id}"
        thread_states = context.user_data.setdefault("thread_state", {})
        state = thread_states.get(state_key, {})

        project_root = project.absolute_path
        current_dir_raw = state.get("current_directory")
        current_dir = (
            Path(current_dir_raw).resolve() if current_dir_raw else project_root
        )
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        context.user_data["current_directory"] = current_dir
        context.user_data["claude_session_id"] = state.get("claude_session_id")
        context.user_data["_thread_context"] = {
            "chat_id": chat.id,
            "message_thread_id": message_thread_id,
            "state_key": state_key,
            "project_slug": project.slug,
            "project_root": str(project_root),
            "project_name": project.name,
        }
        return True

    def _persist_thread_state(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Persist compatibility keys back into per-thread state."""
        thread_context = context.user_data.get("_thread_context")
        if not thread_context:
            return

        project_root = Path(thread_context["project_root"])
        current_dir = context.user_data.get("current_directory", project_root)
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))
        current_dir = current_dir.resolve()
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        thread_states = context.user_data.setdefault("thread_state", {})
        thread_states[thread_context["state_key"]] = {
            "current_directory": str(current_dir),
            "claude_session_id": context.user_data.get("claude_session_id"),
            "project_slug": thread_context["project_slug"],
        }

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """Return True if path is within root."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_message_thread_id(update: Update) -> Optional[int]:
        """Extract topic/thread id from update message for forum/direct topics."""
        message = update.effective_message
        if not message:
            return None
        message_thread_id = getattr(message, "message_thread_id", None)
        if isinstance(message_thread_id, int) and message_thread_id > 0:
            return message_thread_id
        dm_topic = getattr(message, "direct_messages_topic", None)
        topic_id = getattr(dm_topic, "topic_id", None) if dm_topic else None
        if isinstance(topic_id, int) and topic_id > 0:
            return topic_id
        # Telegram omits message_thread_id for the General topic in forum
        # supergroups; its canonical thread ID is 1.
        chat = update.effective_chat
        if chat and getattr(chat, "is_forum", False):
            return 1
        return None

    async def _reject_for_thread_mode(self, update: Update, message: str) -> None:
        """Send a guidance response when strict thread routing rejects an update."""
        query = update.callback_query
        if query:
            try:
                await query.answer()
            except Exception:
                pass
            if query.message:
                await query.message.reply_text(message, parse_mode="HTML")
            return

        if update.effective_message:
            await update.effective_message.reply_text(message, parse_mode="HTML")

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register agentic handlers: commands + text/file/photo."""
        from .handlers import command

        # Commands
        handlers = [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("status", self.agentic_status),
            ("sessions", self.agentic_sessions),
            ("verbose", self.agentic_verbose),
            ("repo", self.agentic_repo),
            ("restart", command.restart_command),
            ("model", self.agentic_model),
            ("models", self.agentic_model),
            ("mode", self.agentic_mode),
            ("stop", self.agentic_stop),
            ("cancel", self.agentic_stop),
            ("compact", self.agentic_compact),
            ("tokens", self.agentic_tokens),
            ("cost", self.agentic_tokens),
            ("workspace", self.agentic_workspace),
            ("menu", self.agentic_menu),
            ("help", self.agentic_help),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        # Derive known commands dynamically — avoids drift when new commands are added
        self._known_commands: frozenset[str] = frozenset(cmd for cmd, _ in handlers)

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text),
            ),
            group=10,
        )

        # Unknown slash commands -> Claude (passthrough in agentic mode).
        # Registered commands are handled by CommandHandlers in group 0
        # (higher priority). This catches any /command not matched there
        # and forwards it to Claude, while skipping known commands to
        # avoid double-firing.
        app.add_handler(
            MessageHandler(
                filters.COMMAND,
                self._inject_deps(self._handle_unknown_command),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Voice messages -> transcribe -> Claude
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(self.agentic_voice)),
            group=10,
        )

        # Stop button callback (must be before cd: handler)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_stop_callback),
                pattern=r"^stop:",
            )
        )

        # Interactive tool-approval callback (Allow/Deny buttons)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_tool_approval_callback),
                pattern=r"^tapv:",
            )
        )

        # Only cd: callbacks (for project selection), scoped by pattern
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )

        # sess: callbacks (resume a chosen session)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_sessions_callback),
                pattern=r"^sess:",
            )
        )

        # set: callbacks (model, mode, ws)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_set_callback),
                pattern=r"^set:",
            )
        )

        # menu: callbacks (navigation)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_menu_callback),
                pattern=r"^menu:",
            )
        )

        # action: callbacks
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_action_callback),
                pattern=r"^action:",
            )
        )

        # noop callback
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_noop_callback),
                pattern=r"^noop$",
            )
        )

        logger.info("Agentic handlers registered")

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
            ("restart", command.restart_command),
            ("model", self.agentic_model),
            ("models", self.agentic_model),
            ("mode", self.agentic_mode),
            ("stop", self.agentic_stop),
            ("cancel", self.agentic_stop),
            ("compact", self.agentic_compact),
            ("tokens", self.agentic_tokens),
            ("cost", self.agentic_tokens),
            ("workspace", self.agentic_workspace),
            ("menu", self.agentic_menu),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(message.handle_voice)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_set_callback),
                pattern=r"^set:",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_menu_callback),
                pattern=r"^menu:",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_action_callback),
                pattern=r"^action:",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_noop_callback),
                pattern=r"^noop$",
            )
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (with Antigravity CLI extensions)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            commands = [
                BotCommand("start", "Khởi động bot"),
                BotCommand("menu", "Bàn phím điều hướng tổng thể"),
                BotCommand("model", "Chọn model Claude (3.7 Sonnet, Haiku, Opus)"),
                BotCommand("mode", "Chế độ plan hoặc accept-edits"),
                BotCommand("workspace", "Xem và chuyển đổi workspace"),
                BotCommand("tokens", "Báo cáo token & chi phí session"),
                BotCommand("compact", "Nén ngữ cảnh session"),
                BotCommand("stop", "Ngắt tác vụ đang xử lý"),
                BotCommand("cancel", "Hủy tác vụ đang xử lý"),
                BotCommand("new", "Bắt đầu phiên làm việc mới"),
                BotCommand("sessions", "Danh sách & tiếp tục session"),
                BotCommand("status", "Xem trạng thái hệ thống"),
                BotCommand("help", "Xem hướng dẫn lệnh"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands
        else:
            commands = [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands

    # --- Agentic handlers ---

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome, no buttons."""
        user = update.effective_user
        sync_line = ""
        if (
            self.settings.enable_project_threads
            and self.settings.project_threads_mode == "private"
        ):
            if (
                not update.effective_chat
                or getattr(update.effective_chat, "type", "") != "private"
            ):
                await update.message.reply_text(
                    "🚫 <b>Private Topics Mode</b>\n\n"
                    "Use this bot in a private chat and run <code>/start</code> there.",
                    parse_mode="HTML",
                )
                return
            manager = context.bot_data.get("project_threads_manager")
            if manager:
                try:
                    result = await manager.sync_topics(
                        context.bot,
                        chat_id=update.effective_chat.id,
                    )
                    sync_line = (
                        "\n\n🧵 Topics synced"
                        f" (created {result.created}, reused {result.reused})."
                    )
                except PrivateTopicsUnavailableError:
                    await update.message.reply_text(
                        manager.private_topics_unavailable_message(),
                        parse_mode="HTML",
                    )
                    return
                except Exception:
                    sync_line = "\n\n🧵 Topic sync failed. Run /sync_threads to retry."
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = f"<code>{current_dir}/</code>"

        safe_name = escape_html(user.first_name)
        await update.message.reply_text(
            f"Hi {safe_name}! I'm your AI coding assistant.\n"
            f"Just tell me what you need — I can read, write, and run code.\n\n"
            f"Working in: {dir_display}\n"
            f"Commands: /new (reset) · /status"
            f"{sync_line}",
            parse_mode="HTML",
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        context.user_data["claude_session_id"] = None
        context.user_data["session_started"] = True
        context.user_data["force_new_session"] = True
        context.user_data.pop("session_usage", None)

        await update.message.reply_text("🆕 <b>Đã khởi tạo phiên làm việc mới.</b>\nBạn muốn làm gì tiếp theo?", parse_mode="HTML")

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact one-line status, no buttons."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = str(current_dir)

        session_id = context.user_data.get("claude_session_id")
        session_status = "active" if session_id else "none"

        # Cost info
        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(update.effective_user.id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f" · Cost: ${current_cost:.2f}"
            except Exception:
                pass

        await update.message.reply_text(
            f"📂 {dir_display} · Session: {session_status}{cost_str}"
        )

    async def agentic_sessions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List saved sessions in the current project, with buttons to resume."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        current_dir = Path(current_dir)

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await update.message.reply_text("Session manager unavailable.")
            return

        try:
            all_sessions = await claude_integration.get_user_sessions(
                update.effective_user.id
            )
        except Exception as e:
            logger.warning("Failed to list sessions", error=str(e))
            await update.message.reply_text("Couldn't load sessions.")
            return

        # Keep sessions for the current project that have a real, non-expired id
        sessions = [
            s
            for s in all_sessions
            if Path(s["project_path"]) == current_dir
            and s.get("session_id")
            and not s.get("expired")
        ]
        sessions.sort(key=lambda s: s["last_used"], reverse=True)

        if not sessions:
            await update.message.reply_text(
                f"No saved sessions in <code>{escape_html(current_dir.name)}/</code>.\n"
                "Start chatting to create one.",
                parse_mode="HTML",
            )
            return

        active_id = context.user_data.get("claude_session_id")
        lines: List[str] = [
            f"<b>Sessions in {escape_html(current_dir.name)}/</b>\n"
        ]
        keyboard_rows: List[list] = []  # type: ignore[type-arg]

        for idx, s in enumerate(sessions, start=1):
            try:
                last = _format_relative_time(datetime.fromisoformat(s["last_used"]))
            except (ValueError, KeyError):
                last = "?"
            msgs = s.get("message_count", 0)
            cost = s.get("total_cost", 0.0)
            marker = " ◀" if s["session_id"] == active_id else ""
            lines.append(
                f"{idx}. {last} · {msgs} msgs · ${cost:.2f}{marker}"
            )
            keyboard_rows.append(
                [
                    InlineKeyboardButton(
                        f"{idx}. {last} · {msgs} msgs",
                        callback_data=f"sess:{s['session_id']}",
                    )
                ]
            )

        await update.message.reply_text(
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard_rows),
        )

    async def _agentic_sessions_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle sess: callbacks — resume the chosen session."""
        query = update.callback_query
        await query.answer()

        session_id = query.data.split(":", 1)[1]

        # Verify the session belongs to this user before resuming
        claude_integration = context.bot_data.get("claude_integration")
        owned = False
        if claude_integration:
            try:
                user_sessions = await claude_integration.get_user_sessions(
                    query.from_user.id
                )
                owned = any(s.get("session_id") == session_id for s in user_sessions)
            except Exception:
                owned = False

        if not owned:
            await query.edit_message_text("That session is no longer available.")
            return

        context.user_data["claude_session_id"] = session_id
        context.user_data["force_new_session"] = False

        await query.edit_message_text(
            f"✅ Resumed session <code>{escape_html(session_id[:8])}</code>. "
            "Continue where you left off.",
            parse_mode="HTML",
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="sessions",
                args=["resume"],
                success=True,
            )

    def _get_verbose_level(self, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return effective verbose level: per-user override or global default."""
        user_override = context.user_data.get("verbose_level")
        if user_override is not None:
            return int(user_override)
        return self.settings.verbose_level

    async def agentic_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set output verbosity: /verbose [0|1|2]."""
        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            current = self._get_verbose_level(context)
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await update.message.reply_text(
                f"Verbosity: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Usage: <code>/verbose 0|1|2</code>\n"
                "  0 = quiet (final response only)\n"
                "  1 = normal (tools + reasoning)\n"
                "  2 = detailed (tools with inputs + reasoning)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Please use: /verbose 0, /verbose 1, or /verbose 2"
            )
            return

        context.user_data["verbose_level"] = level
        labels = {0: "quiet", 1: "normal", 2: "detailed"}
        await update.message.reply_text(
            f"Verbosity set to <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    def _format_verbose_progress(
        self,
        activity_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
    ) -> str:
        """Build the progress message text based on activity so far."""
        if not activity_log:
            return "Working..."

        elapsed = time.time() - start_time
        lines: List[str] = [f"Working... ({elapsed:.0f}s)\n"]

        for entry in activity_log[-15:]:  # Show last 15 entries max
            kind = entry.get("kind", "tool")
            if kind == "text":
                # Claude's intermediate reasoning/commentary
                snippet = entry.get("detail", "")
                if verbose_level >= 2:
                    lines.append(f"\U0001f4ac {snippet}")
                else:
                    # Level 1: one short line
                    lines.append(f"\U0001f4ac {snippet[:80]}")
            else:
                # Tool call
                icon = _tool_icon(entry["name"])
                if verbose_level >= 2 and entry.get("detail"):
                    lines.append(f"{icon} {entry['name']}: {entry['detail']}")
                else:
                    lines.append(f"{icon} {entry['name']}")

        if len(activity_log) > 15:
            lines.insert(1, f"... ({len(activity_log) - 15} earlier entries)\n")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a short summary of tool input for verbose level 2."""
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                # Show just the filename, not the full path
                return path.rsplit("/", 1)[-1]
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return pattern[:60]
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                return _redact_secrets(cmd[:100])[:80]
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:60]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:60]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""

    @staticmethod
    def _summarize_tool_input_for_approval(
        tool_name: str, tool_input: Dict[str, Any]
    ) -> str:
        """Return a detailed summary of tool input for an approval prompt.

        Unlike ``_summarize_tool_input`` (built for compact verbose-log
        lines), this needs to give a human enough detail to make an
        allow/deny security decision: the full file path rather than just
        the filename, and a long window for Bash commands rather than an
        80-character preview that hides everything after it.
        """
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                return path
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return pattern[:200]
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                redacted = _redact_secrets(cmd)
                if len(redacted) > 1000:
                    return redacted[:1000] + "…"
                return redacted
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:200]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:200]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:200]
        return ""

    @staticmethod
    def _start_typing_heartbeat(
        chat: Any,
        interval: float = 2.0,
    ) -> "asyncio.Task[None]":
        """Start a background typing indicator task.

        Sends typing every *interval* seconds, independently of
        stream events. Cancel the returned task in a ``finally``
        block.
        """

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await chat.send_action("typing")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_heartbeat())

    def _make_stream_callback(
        self,
        verbose_level: int,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        start_time: float,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
        mcp_images: Optional[List[ImageAttachment]] = None,
        approved_directory: Optional[Path] = None,
        draft_streamer: Optional[DraftStreamer] = None,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> Optional[Callable[[StreamUpdate], Any]]:
        """Create a stream callback for verbose progress updates.

        When *mcp_images* is provided, the callback also intercepts
        ``send_image_to_user`` tool calls and collects validated
        :class:`ImageAttachment` objects for later Telegram delivery.

        When *draft_streamer* is provided, tool activity and assistant
        text are streamed to the user in real time via
        ``sendMessageDraft``.

        Returns None when verbose_level is 0 **and** no MCP image
        collection or draft streaming is requested.
        Typing indicators are handled by a separate heartbeat task.
        """
        need_mcp_intercept = mcp_images is not None and approved_directory is not None

        if verbose_level == 0 and not need_mcp_intercept and draft_streamer is None:
            return None

        last_edit_time = [0.0]  # mutable container for closure

        async def _on_stream(update_obj: StreamUpdate) -> None:
            # Stop all streaming activity after interrupt
            if interrupt_event is not None and interrupt_event.is_set():
                return

            # Intercept send_image_to_user MCP tool calls.
            # The SDK namespaces MCP tools as "mcp__<server>__<tool>",
            # so match both the bare name and the namespaced variant.
            if update_obj.tool_calls and need_mcp_intercept:
                for tc in update_obj.tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name == "send_image_to_user" or tc_name.endswith(
                        "__send_image_to_user"
                    ):
                        tc_input = tc.get("input", {})
                        file_path = tc_input.get("file_path", "")
                        caption = tc_input.get("caption", "")
                        img = validate_image_path(
                            file_path, approved_directory, caption
                        )
                        if img:
                            mcp_images.append(img)

            # Capture tool calls
            if update_obj.tool_calls:
                for tc in update_obj.tool_calls:
                    name = tc.get("name", "unknown")
                    detail = self._summarize_tool_input(name, tc.get("input", {}))
                    if verbose_level >= 1:
                        tool_log.append(
                            {"kind": "tool", "name": name, "detail": detail}
                        )
                    if draft_streamer:
                        icon = _tool_icon(name)
                        line = (
                            f"{icon} {name}: {detail}" if detail else f"{icon} {name}"
                        )
                        await draft_streamer.append_tool(line)

            # Capture assistant text (reasoning / commentary)
            if update_obj.type == "assistant" and update_obj.content:
                text = update_obj.content.strip()
                if text:
                    first_line = text.split("\n", 1)[0].strip()
                    if first_line:
                        if verbose_level >= 1:
                            tool_log.append(
                                {"kind": "text", "detail": first_line[:120]}
                            )
                        if draft_streamer:
                            await draft_streamer.append_tool(
                                f"\U0001f4ac {first_line[:120]}"
                            )

            # Stream text to user via draft (prefer token deltas;
            # skip full assistant messages to avoid double-appending)
            if draft_streamer and update_obj.content:
                if update_obj.type == "stream_delta":
                    await draft_streamer.append_text(update_obj.content)

            # Throttle progress message edits to avoid Telegram rate limits
            if not draft_streamer and verbose_level >= 1:
                now = time.time()
                if (now - last_edit_time[0]) >= 2.0 and tool_log:
                    last_edit_time[0] = now
                    new_text = self._format_verbose_progress(
                        tool_log, verbose_level, start_time
                    )
                    try:
                        await progress_msg.edit_text(
                            new_text, reply_markup=reply_markup
                        )
                    except Exception:
                        pass

        return _on_stream

    async def _send_images(
        self,
        update: Update,
        images: List[ImageAttachment],
        reply_to_message_id: Optional[int] = None,
        caption: Optional[str] = None,
        caption_parse_mode: Optional[str] = None,
    ) -> bool:
        """Send extracted images as a media group (album) or documents.

        If *caption* is provided and fits (≤1024 chars), it is attached to the
        photo / first album item so text + images appear as one message.

        Returns True if the caption was successfully embedded in the photo message.
        """
        photos: List[ImageAttachment] = []
        documents: List[ImageAttachment] = []
        for img in images:
            if should_send_as_photo(img.path):
                photos.append(img)
            else:
                documents.append(img)

        # Telegram caption limit
        use_caption = bool(
            caption and len(caption) <= 1024 and photos and not documents
        )
        caption_sent = False

        # Send raster photos as a single album (Telegram groups 2-10 items)
        if photos:
            try:
                if len(photos) == 1:
                    with open(photos[0].path, "rb") as f:
                        await update.message.reply_photo(
                            photo=f,
                            reply_to_message_id=reply_to_message_id,
                            caption=caption if use_caption else None,
                            parse_mode=caption_parse_mode if use_caption else None,
                        )
                    caption_sent = use_caption
                else:
                    media = []
                    file_handles = []
                    for idx, img in enumerate(photos[:10]):
                        fh = open(img.path, "rb")  # noqa: SIM115
                        file_handles.append(fh)
                        media.append(
                            InputMediaPhoto(
                                media=fh,
                                caption=caption if use_caption and idx == 0 else None,
                                parse_mode=(
                                    caption_parse_mode
                                    if use_caption and idx == 0
                                    else None
                                ),
                            )
                        )
                    try:
                        await update.message.chat.send_media_group(
                            media=media,
                            reply_to_message_id=reply_to_message_id,
                        )
                        caption_sent = use_caption
                    finally:
                        for fh in file_handles:
                            fh.close()
            except Exception as e:
                logger.warning("Failed to send photo album", error=str(e))

        # Send SVGs / large files as documents (one by one — can't mix in album)
        for img in documents:
            try:
                with open(img.path, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=img.path.name,
                        reply_to_message_id=reply_to_message_id,
                    )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(
                    "Failed to send document image",
                    path=str(img.path),
                    error=str(e),
                )

        return caption_sent

    async def agentic_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Direct Claude passthrough. Simple progress. No suggestions."""
        user_id = update.effective_user.id
        message_text = update.message.text

        logger.info(
            "Agentic text message",
            user_id=user_id,
            message_length=len(message_text),
        )

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await update.message.reply_text(f"⏱️ {limit_message}")
                return

        chat = update.message.chat
        await chat.send_action("typing")

        verbose_level = self._get_verbose_level(context)

        # Create Stop button and interrupt event
        interrupt_event = asyncio.Event()
        stop_kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Stop", callback_data=f"stop:{user_id}")]]
        )
        progress_msg = await update.message.reply_text(
            "Working...", reply_markup=stop_kb
        )

        # Register active request for stop callback
        active_request = ActiveRequest(
            user_id=user_id,
            interrupt_event=interrupt_event,
            progress_msg=progress_msg,
        )
        self._active_requests[user_id] = active_request

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            self._active_requests.pop(user_id, None)
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration.",
                reply_markup=None,
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        # --- Verbose progress tracking via stream callback ---
        tool_log: List[Dict[str, Any]] = []
        start_time = time.time()
        mcp_images: List[ImageAttachment] = []

        # Stream drafts (private chats only)
        draft_streamer: Optional[DraftStreamer] = None
        if self.settings.enable_stream_drafts and chat.type == "private":
            draft_streamer = DraftStreamer(
                bot=context.bot,
                chat_id=chat.id,
                draft_id=generate_draft_id(),
                message_thread_id=update.message.message_thread_id,
                throttle_interval=self.settings.stream_draft_interval,
            )

        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            start_time,
            reply_markup=stop_kb,
            mcp_images=mcp_images,
            approved_directory=self.settings.approved_directory,
            draft_streamer=draft_streamer,
            interrupt_event=interrupt_event,
        )

        # Independent typing heartbeat — stays alive even with no stream events
        heartbeat = self._start_typing_heartbeat(chat)

        approval_cb = None
        if self.settings.interactive_tool_approval:
            approval_cb = self._make_tool_approval_callback(
                user_id=user_id,
                chat_id=chat.id,
                bot=context.bot,
                message_thread_id=update.message.message_thread_id,
            )

        success = True
        try:
            claude_response = await claude_integration.run_command(
                prompt=message_text,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                interrupt_event=interrupt_event,
                approval_callback=approval_cb,
                model=context.user_data.get("claude_model"),
                permission_mode=context.user_data.get("claude_mode"),
            )

            # New session created successfully — clear the one-shot flag
            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id
            self._accumulate_usage(context, claude_response)

            # Track directory changes
            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            # Store interaction
            storage = context.bot_data.get("storage")
            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=message_text,
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log interaction", error=str(e))

            # Format response (no reply_markup — strip keyboards)
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)

            response_content = claude_response.content
            if claude_response.interrupted:
                response_content = (
                    response_content or ""
                ) + "\n\n_(Interrupted by user)_"

            formatted_messages = formatter.format_claude_response(response_content)

        except Exception as e:
            success = False
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(_format_error_message(e), parse_mode="HTML")
            ]
        finally:
            heartbeat.cancel()
            self._active_requests.pop(user_id, None)
            if draft_streamer:
                try:
                    await draft_streamer.flush()
                except Exception:
                    logger.debug("Draft flush failed in finally block", user_id=user_id)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls)
        images: List[ImageAttachment] = mcp_images

        # Try to combine text + images in one message when possible
        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        # Send text messages (skip if caption was already embedded in photos)
        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                try:
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,  # No keyboards in agentic mode
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)
                except Exception as send_err:
                    logger.warning(
                        "Failed to send HTML response, retrying as plain text",
                        error=str(send_err),
                        message_index=i,
                    )
                    try:
                        await update.message.reply_text(
                            message.text,
                            reply_markup=None,
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )
                    except Exception as plain_err:
                        await update.message.reply_text(
                            f"Failed to deliver response "
                            f"(Telegram error: {str(plain_err)[:150]}). "
                            f"Please try again.",
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )

            # Send images separately if caption wasn't used
            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user_id = update.effective_user.id
        document = update.message.document

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await update.message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        if user_id in self._active_requests:
            await update.message.reply_text(
                "⚠️ <b>Đang có một tác vụ đang xử lý.</b>\n\n"
                "Vui lòng chờ hoặc gửi /stop để ngắt tác vụ hiện tại.",
                parse_mode="HTML",
            )
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        interrupt_event = asyncio.Event()
        self._active_requests[user_id] = ActiveRequest(
            user_id=user_id,
            interrupt_event=interrupt_event,
            progress_msg=progress_msg,
        )

        # Try enhanced file handler, fall back to basic
        features = context.bot_data.get("features")
        file_handler = features.get_file_handler() if features else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    update.message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = update.message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_doc: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_doc,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                interrupt_event=interrupt_event,
                model=context.user_data.get("claude_model"),
                permission_mode=context.user_data.get("claude_mode"),
            )

            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id
            self._accumulate_usage(context, claude_response)

            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            try:
                await progress_msg.delete()
            except Exception:
                logger.debug("Failed to delete progress message, ignoring")

            # Use MCP-collected images (from send_image_to_user tool calls)
            images: List[ImageAttachment] = mcp_images_doc

            caption_sent = False
            if images and len(formatted_messages) == 1:
                msg = formatted_messages[0]
                if msg.text and len(msg.text) <= 1024:
                    try:
                        caption_sent = await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                            caption=msg.text,
                            caption_parse_mode=msg.parse_mode,
                        )
                    except Exception as img_err:
                        logger.warning("Image+caption send failed", error=str(img_err))

            if not caption_sent:
                for i, message in enumerate(formatted_messages):
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)

                if images:
                    try:
                        await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                        )
                    except Exception as img_err:
                        logger.warning("Image send failed", error=str(img_err))

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error("Claude file processing failed", error=str(e), user_id=user_id)
        finally:
            heartbeat.cancel()
            self._active_requests.pop(user_id, None)

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        image_handler = features.get_image_handler() if features else None

        if not image_handler:
            await update.message.reply_text("Photo processing is not available.")
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        try:
            photo = update.message.photo[-1]
            processed_image = await image_handler.process_image(
                photo, update.message.caption
            )
            fmt = processed_image.metadata.get("format", "png")
            images = [
                {
                    "data": processed_image.base64_data,
                    "media_type": _MEDIA_TYPE_MAP.get(fmt, "image/png"),
                }
            ]

            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_image.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
                images=images,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude photo processing failed", error=str(e), user_id=user_id
            )

    async def agentic_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Transcribe voice message -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        voice_handler = features.get_voice_handler() if features else None

        if not voice_handler:
            await update.message.reply_text(self._voice_unavailable_message())
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Transcribing...")

        try:
            voice = update.message.voice
            processed_voice = await voice_handler.process_voice_message(
                voice, update.message.caption
            )

            await progress_msg.edit_text("Working...")
            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_voice.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude voice processing failed", error=str(e), user_id=user_id
            )

    async def _handle_agentic_media_message(
        self,
        *,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prompt: str,
        progress_msg: Any,
        user_id: int,
        chat: Any,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Run a media-derived prompt through Claude and send responses."""
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_media: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_media,
            approved_directory=self.settings.approved_directory,
        )

        interrupt_event = asyncio.Event()
        self._active_requests[user_id] = ActiveRequest(
            user_id=user_id,
            interrupt_event=interrupt_event,
            progress_msg=progress_msg,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                interrupt_event=interrupt_event,
                images=images,
                model=context.user_data.get("claude_model"),
                permission_mode=context.user_data.get("claude_mode"),
            )
        finally:
            heartbeat.cancel()
            self._active_requests.pop(user_id, None)

        if force_new:
            context.user_data["force_new_session"] = False

        context.user_data["claude_session_id"] = claude_response.session_id
        self._accumulate_usage(context, claude_response)

        from .handlers.message import _update_working_directory_from_claude_response

        _update_working_directory_from_claude_response(
            claude_response, context, self.settings, user_id
        )

        from .utils.formatting import ResponseFormatter

        formatter = ResponseFormatter(self.settings)
        formatted_messages = formatter.format_claude_response(claude_response.content)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls).
        images: List[ImageAttachment] = mcp_images_media

        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

    async def _handle_unknown_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Forward unknown slash commands to Claude in agentic mode.

        Known commands are handled by their own CommandHandlers (group 0);
        this handler fires for *every* COMMAND message in group 10 but
        returns immediately when the command is registered, preventing
        double execution.
        """
        msg = update.effective_message
        if not msg or not msg.text:
            return
        cmd = msg.text.split()[0].lstrip("/").split("@")[0].lower()
        if cmd in self._known_commands:
            return  # let the registered CommandHandler take care of it
        # Forward unrecognised /commands to Claude as natural language
        await self.agentic_text(update, context)

    def _voice_unavailable_message(self) -> str:
        """Return provider-aware guidance when voice feature is unavailable."""
        if self.settings.voice_provider == "local":
            return (
                "Voice processing is not available. "
                "Ensure whisper.cpp is installed and the model file exists. "
                "Check WHISPER_CPP_BINARY_PATH and WHISPER_CPP_MODEL_PATH settings."
            )
        return (
            "Voice processing is not available. "
            f"Set {self.settings.voice_provider_api_key_env} "
            f"for {self.settings.voice_provider_display_name} and install "
            'voice extras with: pip install "claude-code-telegram[voice]"'
        )

    async def agentic_repo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List repos in workspace or switch to one.

        /repo          — list subdirectories with git indicators
        /repo <name>   — switch to that directory, resume session if available
        """
        args = update.message.text.split()[1:] if update.message.text else []
        base = self.settings.approved_directory
        current_dir = context.user_data.get("current_directory", base)

        if args:
            # Switch to named repo
            target_name = args[0]
            target_path = base / target_name
            if not target_path.is_dir():
                await update.message.reply_text(
                    f"Directory not found: <code>{escape_html(target_name)}</code>",
                    parse_mode="HTML",
                )
                return

            context.user_data["current_directory"] = target_path

            # Try to find a resumable session
            claude_integration = context.bot_data.get("claude_integration")
            session_id = None
            if claude_integration:
                existing = await claude_integration._find_resumable_session(
                    update.effective_user.id, target_path
                )
                if existing:
                    session_id = existing.session_id
            context.user_data["claude_session_id"] = session_id

            is_git = (target_path / ".git").is_dir()
            git_badge = " (git)" if is_git else ""
            session_badge = " · session resumed" if session_id else ""

            await update.message.reply_text(
                f"Switched to <code>{escape_html(target_name)}/</code>"
                f"{git_badge}{session_badge}",
                parse_mode="HTML",
            )
            return

        # No args — list repos
        try:
            entries = sorted(
                [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name,
            )
        except OSError as e:
            await update.message.reply_text(f"Error reading workspace: {e}")
            return

        if not entries:
            await update.message.reply_text(
                f"No repos in <code>{escape_html(str(base))}</code>.\n"
                'Clone one by telling me, e.g. <i>"clone org/repo"</i>.',
                parse_mode="HTML",
            )
            return

        lines: List[str] = []
        keyboard_rows: List[list] = []  # type: ignore[type-arg]
        current_name = current_dir.name if current_dir != base else None

        for d in entries:
            is_git = (d / ".git").is_dir()
            icon = "\U0001f4e6" if is_git else "\U0001f4c1"
            marker = " \u25c0" if d.name == current_name else ""
            lines.append(f"{icon} <code>{escape_html(d.name)}/</code>{marker}")

        # Build inline keyboard (2 per row)
        for i in range(0, len(entries), 2):
            row = []
            for j in range(2):
                if i + j < len(entries):
                    name = entries[i + j].name
                    row.append(InlineKeyboardButton(name, callback_data=f"cd:{name}"))
            keyboard_rows.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard_rows)

        await update.message.reply_text(
            "<b>Repos</b>\n\n" + "\n".join(lines),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def _handle_stop_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle stop: callbacks — interrupt a running Claude request."""
        query = update.callback_query
        target_user_id = int(query.data.split(":", 1)[1])

        # Only the requesting user can stop their own request
        if query.from_user.id != target_user_id:
            await query.answer(
                "Only the requesting user can stop this.", show_alert=True
            )
            return

        active = self._active_requests.get(target_user_id)
        if not active:
            await query.answer("Already completed.", show_alert=False)
            return
        if active.interrupted:
            await query.answer("Already stopping...", show_alert=False)
            return

        active.interrupt_event.set()
        active.interrupted = True
        await query.answer("Stopping...", show_alert=False)

        try:
            await active.progress_msg.edit_text("Stopping...", reply_markup=None)
        except Exception:
            pass

    def _make_tool_approval_callback(
        self,
        user_id: int,
        chat_id: int,
        bot: Any,
        message_thread_id: Optional[int],
    ) -> Callable[[str, Dict[str, Any]], Awaitable[bool]]:
        """Build an approval_callback closure for a single Claude run.

        Sends a Telegram Allow/Deny prompt for a tool call and blocks (with a
        timeout) until the user responds via the ``tapv:`` callback handler.
        Fails closed (denies) on timeout.
        """

        async def request_approval(tool_name: str, tool_input: Dict[str, Any]) -> bool:
            request_id = uuid.uuid4().hex[:12]
            summary = self._summarize_tool_input_for_approval(tool_name, tool_input)
            text = f"⚠️ Claude wants to run <b>{escape_html(tool_name)}</b>"
            if summary:
                text += f"\n<code>{escape_html(summary)}</code>"
            text += "\n\nAllow this action?"

            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Allow", callback_data=f"tapv:allow:{request_id}"
                        ),
                        InlineKeyboardButton(
                            "❌ Deny", callback_data=f"tapv:deny:{request_id}"
                        ),
                    ]
                ]
            )

            # Register the pending approval *before* sending the prompt, so a
            # click that races the send (fast tapper, slow network) always
            # finds an entry instead of hitting "Already handled." and
            # stalling the future until timeout.
            future: "asyncio.Future[bool]" = asyncio.get_running_loop().create_future()
            self._pending_tool_approvals[request_id] = PendingToolApproval(
                user_id=user_id, future=future
            )

            try:
                msg = await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                    message_thread_id=message_thread_id,
                )
            except Exception:
                self._pending_tool_approvals.pop(request_id, None)
                raise

            try:
                return await asyncio.wait_for(
                    future,
                    timeout=self.settings.interactive_tool_approval_timeout_seconds,
                )
            except asyncio.TimeoutError:
                timeout_allow = (
                    self.settings.interactive_tool_approval_timeout_action == "allow"
                )
                try:
                    suffix = (
                        "\n\n⏱ Timed out — auto-allowed"
                        if timeout_allow
                        else "\n\n⏱ Timed out — denied"
                    )
                    await msg.edit_text(
                        text + suffix,
                        parse_mode="HTML",
                        reply_markup=None,
                    )
                except Exception:
                    pass
                return timeout_allow
            finally:
                self._pending_tool_approvals.pop(request_id, None)

        return request_approval

    async def _handle_tool_approval_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle tapv: callbacks — resolve a pending tool-approval prompt."""
        query = update.callback_query
        _, action, request_id = query.data.split(":", 2)

        pending = self._pending_tool_approvals.get(request_id)
        if pending is None:
            await query.answer("Already handled.", show_alert=False)
            return

        if query.from_user.id != pending.user_id:
            await query.answer("Only the requesting user can respond.", show_alert=True)
            return

        if pending.future.done():
            await query.answer("Already handled.", show_alert=False)
            return

        approved = action == "allow"
        pending.future.set_result(approved)
        await query.answer("Allowed" if approved else "Denied", show_alert=False)

        try:
            status = "✅ Allowed" if approved else "❌ Denied"
            await query.edit_message_text(status, reply_markup=None)
        except Exception:
            pass

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks — switch directory and resume session if available."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, project_name = data.split(":", 1)

        base = self.settings.approved_directory
        new_path = base / project_name

        if not new_path.is_dir():
            await query.edit_message_text(
                f"Directory not found: <code>{escape_html(project_name)}</code>",
                parse_mode="HTML",
            )
            return

        context.user_data["current_directory"] = new_path

        # Look for a resumable session instead of always clearing
        claude_integration = context.bot_data.get("claude_integration")
        session_id = None
        if claude_integration:
            existing = await claude_integration._find_resumable_session(
                query.from_user.id, new_path
            )
            if existing:
                session_id = existing.session_id
        context.user_data["claude_session_id"] = session_id

        is_git = (new_path / ".git").is_dir()
        git_badge = " (git)" if is_git else ""
        session_badge = " · session resumed" if session_id else ""

        await query.edit_message_text(
            f"Switched to <code>{escape_html(project_name)}/</code>"
            f"{git_badge}{session_badge}",
            parse_mode="HTML",
        )

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="cd",
                args=[project_name],
                success=True,
            )

    def _accumulate_usage(
        self, context: ContextTypes.DEFAULT_TYPE, claude_response: Any
    ) -> None:
        """Accumulate token usage and cost in context.user_data."""
        stats = context.user_data.setdefault(
            "session_usage",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "total_cost": 0.0,
                "turns": 0,
            },
        )
        usage = getattr(claude_response, "usage", None)
        if isinstance(usage, dict):
            stats["input_tokens"] += usage.get("input_tokens", 0) or 0
            stats["output_tokens"] += usage.get("output_tokens", 0) or 0
            stats["cache_creation_tokens"] += (
                usage.get("cache_creation_input_tokens", 0) or 0
            )
            stats["cache_read_tokens"] += (
                usage.get("cache_read_input_tokens", 0) or 0
            )
        stats["total_cost"] += getattr(claude_response, "cost", 0.0) or 0.0
        stats["turns"] += getattr(claude_response, "num_turns", 1) or 1

    async def agentic_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /model command - view or select Claude model."""
        msg = update.effective_message
        if not msg:
            return
        args = msg.text.split()[1:] if msg.text else []
        models = await fetch_claude_models()
        if args:
            target = args[0].strip()
            if target.lower() in ("default", "clear", "reset"):
                context.user_data["claude_model"] = None
                await msg.reply_text(
                    "🔄 <b>Đã đặt lại model về mặc định hệ thống.</b>",
                    parse_mode="HTML",
                    reply_markup=model_keyboard(None, models=models),
                )
                return

            chosen_model = resolve_model_id(target, models=models)
            context.user_data["claude_model"] = chosen_model
            await msg.reply_text(
                f"🤖 <b>Đã chuyển sang model:</b> <code>{escape_html(chosen_model or '')}</code>",
                parse_mode="HTML",
                reply_markup=model_keyboard(chosen_model, models=models),
            )
            return

        current_model = context.user_data.get("claude_model")
        display_name = current_model or "Mặc định hệ thống"
        await msg.reply_text(
            f"🤖 <b>Chọn Model Claude:</b>\n\nModel hiện tại: <code>{escape_html(display_name)}</code>",
            parse_mode="HTML",
            reply_markup=model_keyboard(current_model, models=models),
        )

    async def agentic_mode(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /mode command - switch between plan, accept-edits, default."""
        msg = update.effective_message
        if not msg:
            return
        args = msg.text.split()[1:] if msg.text else []
        if args:
            target = args[0].strip().lower().replace("-", "").replace("_", "")
            if target in ("plan",):
                context.user_data["claude_mode"] = "plan"
                await msg.reply_text(
                    "📋 <b>Đã chuyển sang chế độ: Plan (Chỉ lập kế hoạch, không sửa code)</b>",
                    parse_mode="HTML",
                    reply_markup=mode_keyboard("plan"),
                )
                return
            elif target in ("acceptedits", "edit", "edits"):
                context.user_data["claude_mode"] = "acceptEdits"
                await msg.reply_text(
                    "⚡ <b>Đã chuyển sang chế độ: Accept Edits (Tự động thực thi & chỉnh sửa)</b>",
                    parse_mode="HTML",
                    reply_markup=mode_keyboard("acceptEdits"),
                )
                return
            elif target in ("default", "clear", "reset"):
                context.user_data["claude_mode"] = "default"
                await msg.reply_text(
                    "🛡️ <b>Đã chuyển sang chế độ: Mặc định (Hỏi xác nhận khi cần)</b>",
                    parse_mode="HTML",
                    reply_markup=mode_keyboard("default"),
                )
                return
            else:
                await msg.reply_text(
                    f"⚠️ Chế độ không hợp lệ: <code>{escape_html(args[0])}</code>\n"
                    "Lựa chọn: <code>plan</code>, <code>accept-edits</code>, <code>default</code>",
                    parse_mode="HTML",
                    reply_markup=mode_keyboard(context.user_data.get("claude_mode")),
                )
                return

        current_mode = context.user_data.get("claude_mode", "default")
        await msg.reply_text(
            "⚙️ <b>Chế độ thực thi (Execution Mode):</b>\n\n"
            "• 📋 <b>Plan</b>: Lập kế hoạch chi tiết, không chỉnh sửa file.\n"
            "• ⚡ <b>Accept Edits</b>: Tự động chạy lệnh và áp dụng sửa đổi mã nguồn.\n"
            "• 🛡️ <b>Default</b>: Chế độ an toàn chuẩn (hỏi xác nhận khi cần).\n\n"
            f"Chế độ hiện tại: <code>{escape_html(current_mode)}</code>",
            parse_mode="HTML",
            reply_markup=mode_keyboard(current_mode),
        )

    async def agentic_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /stop or /cancel command - interrupt running request."""
        user = update.effective_user
        msg = update.effective_message
        if not user or not msg:
            return
        active = self._active_requests.get(user.id)
        if not active:
            await msg.reply_text(
                "ℹ️ <b>Hiện không có tác vụ nào đang xử lý.</b>",
                parse_mode="HTML",
            )
            return

        if active.interrupted:
            await msg.reply_text(
                "⏳ <b>Tác vụ đang trong quá trình dừng...</b>",
                parse_mode="HTML",
            )
            return

        active.interrupt_event.set()
        active.interrupted = True
        try:
            if active.progress_msg:
                await active.progress_msg.edit_text(
                    "🛑 Đang dừng tác vụ...", reply_markup=None
                )
        except Exception:
            pass

        await msg.reply_text(
            "🛑 <b>Đã gửi lệnh ngắt tác vụ đang xử lý.</b>",
            parse_mode="HTML",
        )

    async def agentic_compact(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /compact command - trigger context compaction."""
        user = update.effective_user
        msg = update.effective_message
        if not user or not msg:
            return

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await msg.reply_text("Claude integration not available.")
            return

        session_id = context.user_data.get("claude_session_id")
        if not session_id:
            await msg.reply_text(
                "ℹ️ Chưa có phiên làm việc nào đang mở để nén ngữ cảnh.\n"
                "Hãy gửi tin nhắn bất kỳ để bắt đầu một phiên mới.",
                parse_mode="HTML",
            )
            return

        if self._active_requests.get(user.id):
            await msg.reply_text(
                "⚠️ Đang có tác vụ khác đang chạy. Vui lòng chờ hoặc gửi /stop trước.",
                parse_mode="HTML",
            )
            return

        if update.callback_query:
            compact_prompt = "/compact"
        else:
            args = (
                msg.text.split(maxsplit=1)[1]
                if msg and msg.text and len(msg.text.split()) > 1
                else ""
            )
            compact_prompt = f"/compact {args}".strip()

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        progress_msg = await msg.reply_text(
            "🗜️ <i>Đang thực hiện nén ngữ cảnh phiên làm việc...</i>",
            parse_mode="HTML",
        )

        interrupt_event = asyncio.Event()
        self._active_requests[user.id] = ActiveRequest(
            user_id=user.id,
            interrupt_event=interrupt_event,
            progress_msg=progress_msg,
        )

        try:
            claude_response = await claude_integration.run_command(
                prompt=compact_prompt,
                working_directory=current_dir,
                user_id=user.id,
                session_id=session_id,
                interrupt_event=interrupt_event,
                model=context.user_data.get("claude_model"),
                permission_mode=context.user_data.get("claude_mode"),
            )
            self._accumulate_usage(context, claude_response)
            result_preview = (
                escape_html(claude_response.content[:800])
                if claude_response.content
                else "Đã hoàn thành nén ngữ cảnh."
            )
            await progress_msg.edit_text(
                f"🗜️ <b>Ngữ cảnh đã được nén thành công!</b>\n\n{result_preview}",
                parse_mode="HTML",
                reply_markup=tokens_keyboard(),
            )
        except Exception as e:
            await progress_msg.edit_text(
                f"❌ <b>Nén ngữ cảnh thất bại:</b>\n<code>{escape_html(str(e))}</code>",
                parse_mode="HTML",
            )
        finally:
            self._active_requests.pop(user.id, None)

    async def agentic_tokens(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /tokens or /cost command - report token usage and cost."""
        msg = update.effective_message
        if not msg:
            return

        stats = context.user_data.get(
            "session_usage",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "total_cost": 0.0,
                "turns": 0,
            },
        )
        session_id = context.user_data.get("claude_session_id")
        session_display = (
            f"<code>{escape_html(session_id[:16])}...</code>"
            if session_id
            else "<i>Chưa tạo phiên</i>"
        )
        current_model = (
            context.user_data.get("claude_model")
            or self.settings.claude_model
            or "Mặc định (CLI)"
        )
        current_mode = context.user_data.get("claude_mode") or "default"

        total_tokens = (
            stats.get("input_tokens", 0)
            + stats.get("output_tokens", 0)
            + stats.get("cache_read_tokens", 0)
            + stats.get("cache_creation_tokens", 0)
        )

        text = (
            "📊 <b>Báo Cáo Token & Chi Phí Phiên Làm Việc</b>\n\n"
            f"🆔 <b>Session:</b> {session_display}\n"
            f"🤖 <b>Model:</b> <code>{escape_html(current_model)}</code>\n"
            f"⚙️ <b>Chế độ:</b> <code>{escape_html(current_mode)}</code>\n"
            f"🔄 <b>Lượt tương tác (Turns):</b> <code>{stats.get('turns', 0)}</code>\n\n"
            "🔢 <b>Thống kê Token:</b>\n"
            f"• <b>Prompt (Input):</b> <code>{stats.get('input_tokens', 0):,}</code>\n"
            f"• <b>Completion (Output):</b> <code>{stats.get('output_tokens', 0):,}</code>\n"
            f"• <b>Cache Đọc:</b> <code>{stats.get('cache_read_tokens', 0):,}</code>\n"
            f"• <b>Cache Tạo:</b> <code>{stats.get('cache_creation_tokens', 0):,}</code>\n"
            f"• <b>Tổng Tokens:</b> <b>{total_tokens:,}</b>\n\n"
            f"💰 <b>Chi phí ước tính:</b> <b>${stats.get('total_cost', 0.0):.4f} USD</b>"
        )
        await msg.reply_text(
            text, parse_mode="HTML", reply_markup=tokens_keyboard()
        )

    async def agentic_workspace(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /workspace command - view and switch active workspace."""
        msg = update.effective_message
        if not msg:
            return
        args = msg.text.split()[1:] if msg.text else []
        base = self.settings.approved_directory
        current_dir = context.user_data.get("current_directory", base)

        if args:
            sub = args[0].strip()
            if sub.lower() in ("clear", "reset", "default"):
                context.user_data["current_directory"] = base
                context.user_data["claude_session_id"] = None
                context.user_data["force_new_session"] = True
                context.user_data.pop("session_usage", None)
                await msg.reply_text(
                    f"🔄 <b>Đã đặt lại thư mục làm việc về mặc định:</b>\n<code>{escape_html(str(base))}</code>\n\nĐã bắt đầu một phiên mới.",
                    parse_mode="HTML",
                    reply_markup=workspace_keyboard(base, base, 0),
                )
                return

            target_path = Path(sub)
            if not target_path.is_absolute():
                target_path = base / sub

            if not target_path.is_dir():
                await msg.reply_text(
                    f"❌ <b>Thư mục không tồn tại:</b> <code>{escape_html(sub)}</code>",
                    parse_mode="HTML",
                )
                return

            try:
                target_path.resolve().relative_to(base.resolve())
            except ValueError:
                await msg.reply_text(
                    "🚫 <b>Thư mục nằm ngoài phạm vi cho phép!</b>",
                    parse_mode="HTML",
                )
                return

            context.user_data["current_directory"] = target_path
            # Check for resumable session
            claude_integration = context.bot_data.get("claude_integration")
            session_id = None
            if claude_integration:
                existing = await claude_integration._find_resumable_session(
                    update.effective_user.id, target_path
                )
                if existing:
                    session_id = existing.session_id
            context.user_data["claude_session_id"] = session_id

            session_badge = (
                " · Đã khôi phục session cũ" if session_id else " · Phiên mới"
            )
            await msg.reply_text(
                f"📁 <b>Đã chuyển thư mục làm việc sang:</b>\n<code>{escape_html(str(target_path))}</code>{session_badge}",
                parse_mode="HTML",
                reply_markup=workspace_keyboard(target_path, base, 0),
            )
            return

        is_custom = current_dir.resolve() != base.resolve()
        text = (
            "📁 <b>Quản Lý Thư Mục Làm Việc (Workspace)</b>\n\n"
            f"📍 <b>Thư mục đang hoạt động:</b> <code>{escape_html(str(current_dir))}</code> {'(tùy chỉnh)' if is_custom else '(mặc định)'}\n"
            f"⚙️ <b>Thư mục gốc:</b> <code>{escape_html(str(base))}</code>\n\n"
            "<b>Cách dùng:</b>\n"
            "• Bấm nút bên dưới để chuyển thư mục nhanh.\n"
            "• Gõ <code>/workspace &lt;tên_thư_mục&gt;</code> để đổi thư mục.\n"
            "• Gõ <code>/workspace reset</code> để về thư mục gốc."
        )
        await msg.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=workspace_keyboard(current_dir, base, 0),
        )

    async def agentic_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /menu command - display master navigation keyboard."""
        msg = update.effective_message
        if not msg:
            return
        text = (
            "⚙️ <b>Bàn Phím Điều Hướng Antigravity CLI</b>\n\n"
            "Chọn chức năng bên dưới để quản lý bot và session của bạn:"
        )
        await msg.reply_text(text, parse_mode="HTML", reply_markup=menu_keyboard())

    async def agentic_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command - show command overview & navigation keyboard."""
        msg = update.effective_message
        if not msg:
            return
        help_text = (
            "🤖 <b>Hướng Dẫn Lệnh Antigravity CLI cho Claude Code:</b>\n\n"
            "▫️ <code>/model</code> - Chọn model AI (3.7 Sonnet, Haiku, Opus)\n"
            "▫️ <code>/mode</code> - Đổi chế độ: <code>plan</code> (kế hoạch) / <code>accept-edits</code> (thực thi)\n"
            "▫️ <code>/workspace</code> - Xem & đổi thư mục làm việc trực tiếp\n"
            "▫️ <code>/tokens</code> | <code>/cost</code> - Báo cáo token và chi phí session\n"
            "▫️ <code>/compact</code> - Nén ngữ cảnh hội thoại\n"
            "▫️ <code>/stop</code> | <code>/cancel</code> - Ngắt ngay tác vụ đang chạy\n"
            "▫️ <code>/new</code> - Khởi tạo phiên làm việc sạch mới\n"
            "▫️ <code>/sessions</code> - Xem & tiếp tục các phiên cũ\n"
            "▫️ <code>/status</code> - Trạng thái hệ thống\n"
            "▫️ <code>/menu</code> - Mở menu điều hướng trực quan\n\n"
            "💡 <i>Bạn cũng có thể gửi văn bản hoặc ảnh để Claude tự động xử lý.</i>"
        )
        await msg.reply_text(help_text, parse_mode="HTML", reply_markup=menu_keyboard())

    async def _handle_set_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle set: callbacks (set:model:, set:mode:, set:ws:)."""
        query = update.callback_query
        if not query or not query.data:
            return
        parts = query.data.split(":")
        category = parts[1] if len(parts) > 1 else ""
        val = parts[2] if len(parts) > 2 else ""

        if category == "model":
            if val == "refresh":
                models = await fetch_claude_models(force_refresh=True)
                cur_model = context.user_data.get("claude_model")
                display_name = cur_model or "Mặc định hệ thống"
                await query.answer(f"Đã làm mới ({len(models)} models)")
                try:
                    await query.edit_message_text(
                        f"🤖 <b>Chọn Model Claude:</b>\n\nModel hiện tại: <code>{escape_html(display_name)}</code>",
                        parse_mode="HTML",
                        reply_markup=model_keyboard(cur_model, models=models, page=0),
                    )
                except Exception:
                    pass
                return

            if val == "default":
                context.user_data["claude_model"] = None
                await query.answer("Đã chọn model mặc định.")
            else:
                context.user_data["claude_model"] = val
                await query.answer(f"Đã chọn: {val}")
            cur_model = context.user_data.get("claude_model")
            display_name = cur_model or "Mặc định hệ thống"
            try:
                models = get_cached_models()
                await query.edit_message_text(
                    f"🤖 <b>Chọn Model Claude:</b>\n\nModel hiện tại: <code>{escape_html(display_name)}</code>",
                    parse_mode="HTML",
                    reply_markup=model_keyboard(cur_model, models=models),
                )
            except Exception:
                pass

        elif category == "mode":
            context.user_data["claude_mode"] = val
            await query.answer(f"Đã chọn chế độ: {val}")
            try:
                await query.edit_message_text(
                    "⚙️ <b>Chế độ thực thi (Execution Mode):</b>\n\n"
                    "• 📋 <b>Plan</b>: Lập kế hoạch chi tiết, không chỉnh sửa file.\n"
                    "• ⚡ <b>Accept Edits</b>: Tự động chạy lệnh và áp dụng sửa đổi mã nguồn.\n"
                    "• 🛡️ <b>Default</b>: Chế độ an toàn chuẩn (hỏi xác nhận khi cần).\n\n"
                    f"Chế độ hiện tại: <code>{escape_html(val)}</code>",
                    parse_mode="HTML",
                    reply_markup=mode_keyboard(val),
                )
            except Exception:
                pass

        elif category == "ws":
            base = self.settings.approved_directory
            if val == "clear":
                context.user_data["current_directory"] = base
                context.user_data["claude_session_id"] = None
                context.user_data["force_new_session"] = True
                context.user_data.pop("session_usage", None)
                await query.answer("Đã về thư mục gốc.")
                cur_dir = base
            else:
                target_path = base / val
                if target_path.is_dir():
                    context.user_data["current_directory"] = target_path
                    claude_integration = context.bot_data.get("claude_integration")
                    session_id = None
                    if claude_integration:
                        existing = await claude_integration._find_resumable_session(
                            query.from_user.id, target_path
                        )
                        if existing:
                            session_id = existing.session_id
                    context.user_data["claude_session_id"] = session_id
                    await query.answer(f"Đã chuyển sang: {val}")
                    cur_dir = target_path
                else:
                    await query.answer("Thư mục không tồn tại!", show_alert=True)
                    cur_dir = context.user_data.get("current_directory", base)

            is_custom = cur_dir.resolve() != base.resolve()
            text = (
                "📁 <b>Quản Lý Thư Mục Làm Việc (Workspace)</b>\n\n"
                f"📍 <b>Thư mục đang hoạt động:</b> <code>{escape_html(str(cur_dir))}</code> {'(tùy chỉnh)' if is_custom else '(mặc định)'}\n"
                f"⚙️ <b>Thư mục gốc:</b> <code>{escape_html(str(base))}</code>\n\n"
                "<b>Cách dùng:</b>\n"
                "• Bấm nút bên dưới để chuyển thư mục nhanh.\n"
                "• Gõ <code>/workspace &lt;tên_thư_mục&gt;</code> để đổi thư mục.\n"
                "• Gõ <code>/workspace reset</code> để về thư mục gốc."
            )
            try:
                await query.edit_message_text(
                    text,
                    parse_mode="HTML",
                    reply_markup=workspace_keyboard(cur_dir, base, 0),
                )
            except Exception:
                pass

    async def _handle_menu_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle menu: navigation callbacks."""
        query = update.callback_query
        if not query or not query.data:
            return
        await query.answer()
        action = query.data.split(":", 1)[1] if ":" in query.data else ""

        if action == "main":
            try:
                await query.edit_message_text(
                    "⚙️ <b>Bàn Phím Điều Hướng Antigravity CLI</b>\n\n"
                    "Chọn chức năng bên dưới để quản lý bot và session của bạn:",
                    parse_mode="HTML",
                    reply_markup=menu_keyboard(),
                )
            except Exception:
                pass
        elif action == "model":
            cur_model = context.user_data.get("claude_model")
            display_name = cur_model or "Mặc định hệ thống"
            models = await fetch_claude_models()
            try:
                await query.edit_message_text(
                    f"🤖 <b>Chọn Model Claude:</b>\n\nModel hiện tại: <code>{escape_html(display_name)}</code>",
                    parse_mode="HTML",
                    reply_markup=model_keyboard(cur_model, models=models, page=0),
                )
            except Exception:
                pass
        elif action.startswith("model:page:"):
            try:
                page = int(action.split(":")[2])
            except ValueError:
                page = 0
            cur_model = context.user_data.get("claude_model")
            models = get_cached_models()
            try:
                await query.edit_message_reply_markup(
                    reply_markup=model_keyboard(cur_model, models=models, page=page)
                )
            except Exception:
                pass
        elif action == "mode":
            cur_mode = context.user_data.get("claude_mode", "default")
            try:
                await query.edit_message_text(
                    "⚙️ <b>Chế độ thực thi (Execution Mode):</b>\n\n"
                    "• 📋 <b>Plan</b>: Lập kế hoạch chi tiết, không chỉnh sửa file.\n"
                    "• ⚡ <b>Accept Edits</b>: Tự động chạy lệnh và áp dụng sửa đổi mã nguồn.\n"
                    "• 🛡️ <b>Default</b>: Chế độ an toàn chuẩn (hỏi xác nhận khi cần).\n\n"
                    f"Chế độ hiện tại: <code>{escape_html(cur_mode)}</code>",
                    parse_mode="HTML",
                    reply_markup=mode_keyboard(cur_mode),
                )
            except Exception:
                pass
        elif action == "workspace":
            base = self.settings.approved_directory
            cur_dir = context.user_data.get("current_directory", base)
            is_custom = cur_dir.resolve() != base.resolve()
            text = (
                "📁 <b>Quản Lý Thư Mục Làm Việc (Workspace)</b>\n\n"
                f"📍 <b>Thư mục đang hoạt động:</b> <code>{escape_html(str(cur_dir))}</code> {'(tùy chỉnh)' if is_custom else '(mặc định)'}\n"
                f"⚙️ <b>Thư mục gốc:</b> <code>{escape_html(str(base))}</code>\n\n"
                "<b>Cách dùng:</b>\n"
                "• Bấm nút bên dưới để chuyển thư mục nhanh.\n"
                "• Gõ <code>/workspace &lt;tên_thư_mục&gt;</code> để đổi thư mục.\n"
                "• Gõ <code>/workspace reset</code> để về thư mục gốc."
            )
            try:
                await query.edit_message_text(
                    text,
                    parse_mode="HTML",
                    reply_markup=workspace_keyboard(cur_dir, base, 0),
                )
            except Exception:
                pass
        elif action.startswith("ws:"):
            try:
                page = int(action.split(":")[1])
            except ValueError:
                page = 0
            base = self.settings.approved_directory
            cur_dir = context.user_data.get("current_directory", base)
            try:
                await query.edit_message_reply_markup(
                    reply_markup=workspace_keyboard(cur_dir, base, page)
                )
            except Exception:
                pass

    async def _handle_action_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle action: callbacks (compact, tokens, cancel, etc.)."""
        query = update.callback_query
        if not query or not query.data:
            return
        action = query.data.split(":", 1)[1] if ":" in query.data else ""

        if action == "compact":
            await query.answer()
            await self.agentic_compact(update, context)
        elif action == "tokens":
            await query.answer()
            await self.agentic_tokens(update, context)
        elif action == "cancel":
            await self.agentic_stop(update, context)
            await query.answer()
        elif action == "new_session":
            await query.answer("Đang khởi tạo phiên mới...")
            context.user_data["claude_session_id"] = None
            context.user_data["session_started"] = True
            context.user_data["force_new_session"] = True
            context.user_data.pop("session_usage", None)
            await query.message.reply_text(
                "🆕 <b>Đã bắt đầu một phiên làm việc mới!</b>",
                parse_mode="HTML",
            )
        elif action == "sessions":
            await query.answer()
            await self.agentic_sessions(update, context)
        elif action == "help":
            await query.answer()
            await self.agentic_help(update, context)
        else:
            await query.answer()

    async def _handle_noop_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle noop callbacks (informational button clicks)."""
        if update.callback_query:
            await update.callback_query.answer()

