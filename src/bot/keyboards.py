"""Inline keyboard builders for Antigravity-style bot navigation."""

from pathlib import Path
from typing import List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ..claude.models import get_cached_models

SUPPORTED_MODELS = get_cached_models()

SUPPORTED_MODES = [
    {"id": "plan", "name": "Plan (Lập kế hoạch)", "icon": "📋"},
    {"id": "acceptEdits", "name": "Accept Edits (Thực thi lệnh)", "icon": "⚡"},
    {"id": "default", "name": "Mặc định (Hỏi xác nhận)", "icon": "🛡️"},
]


def model_keyboard(
    current_model: Optional[str] = None,
    models: Optional[List[dict]] = None,
    page: int = 0,
    page_size: int = 5,
) -> InlineKeyboardMarkup:
    """Build dynamic keyboard for model selection fetched from Claude."""
    buttons: List[List[InlineKeyboardButton]] = []
    available = models if models is not None else get_cached_models()

    total_pages = max(1, (len(available) + page_size - 1) // page_size)
    norm_page = max(0, min(page, total_pages - 1))
    slice_models = available[norm_page * page_size : (norm_page + 1) * page_size]

    cur = (current_model or "").lower().strip()

    for m in slice_models:
        m_id = m["id"]
        m_name = m.get("name", m_id)
        m_icon = m.get("icon", "🤖")
        is_selected = (cur == m_id.lower()) or (
            cur and (cur in m_id.lower() or cur in m_name.lower())
        )
        prefix = "✅ " if is_selected else f"{m_icon} "
        buttons.append([
            InlineKeyboardButton(
                f"{prefix}{m_name}",
                callback_data=f"set:model:{m_id}",
            )
        ])

    # Pagination row
    if total_pages > 1:
        nav_row: List[InlineKeyboardButton] = []
        if norm_page > 0:
            nav_row.append(
                InlineKeyboardButton("‹ Trang trước", callback_data=f"menu:model:page:{norm_page - 1}")
            )
        nav_row.append(InlineKeyboardButton(f"{norm_page + 1}/{total_pages}", callback_data="noop"))
        if norm_page < total_pages - 1:
            nav_row.append(
                InlineKeyboardButton("Trang sau ›", callback_data=f"menu:model:page:{norm_page + 1}")
            )
        buttons.append(nav_row)

    # Actions row
    is_default = not current_model or current_model == "default"
    prefix_def = "✅ " if is_default else "🔄 "
    buttons.append([
        InlineKeyboardButton(
            f"{prefix_def}Mặc định hệ thống",
            callback_data="set:model:default",
        ),
        InlineKeyboardButton(
            "🔄 Làm mới từ Claude",
            callback_data="set:model:refresh",
        ),
    ])
    buttons.append([InlineKeyboardButton("‹ Quay lại Menu", callback_data="menu:main")])
    return InlineKeyboardMarkup(buttons)


def mode_keyboard(current_mode: Optional[str] = None) -> InlineKeyboardMarkup:
    """Build keyboard for execution mode selection."""
    buttons: List[List[InlineKeyboardButton]] = []
    cur = (current_mode or "default").lower().replace("-", "")
    
    for m in SUPPORTED_MODES:
        m_id = m["id"]
        is_selected = cur == m_id.lower().replace("-", "")
        prefix = "✅ " if is_selected else f"{m['icon']} "
        buttons.append([
            InlineKeyboardButton(
                f"{prefix}{m['name']}",
                callback_data=f"set:mode:{m_id}",
            )
        ])
        
    buttons.append([InlineKeyboardButton("‹ Quay lại Menu", callback_data="menu:main")])
    return InlineKeyboardMarkup(buttons)


def workspace_keyboard(
    current_path: Path,
    base_path: Path,
    page: int = 0,
    page_size: int = 6,
) -> InlineKeyboardMarkup:
    """Build keyboard for workspace directories."""
    try:
        entries = sorted(
            [
                d
                for d in base_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ],
            key=lambda d: d.name.lower(),
        )
    except OSError:
        entries = []

    total_pages = max(1, (len(entries) + page_size - 1) // page_size)
    norm_page = max(0, min(page, total_pages - 1))
    slice_entries = entries[norm_page * page_size : (norm_page + 1) * page_size]

    buttons: List[List[InlineKeyboardButton]] = []

    # 2 buttons per row
    for i in range(0, len(slice_entries), 2):
        row: List[InlineKeyboardButton] = []
        d1 = slice_entries[i]
        is_active1 = current_path.resolve() == d1.resolve()
        is_git1 = (d1 / ".git").is_dir()
        icon1 = "✅ " if is_active1 else ("📦 " if is_git1 else "📁 ")
        row.append(InlineKeyboardButton(f"{icon1}{d1.name}", callback_data=f"set:ws:{d1.name}"))

        if i + 1 < len(slice_entries):
            d2 = slice_entries[i + 1]
            is_active2 = current_path.resolve() == d2.resolve()
            is_git2 = (d2 / ".git").is_dir()
            icon2 = "✅ " if is_active2 else ("📦 " if is_git2 else "📁 ")
            row.append(InlineKeyboardButton(f"{icon2}{d2.name}", callback_data=f"set:ws:{d2.name}"))
        buttons.append(row)

    # Pagination row
    if total_pages > 1:
        nav_row: List[InlineKeyboardButton] = []
        if norm_page > 0:
            nav_row.append(InlineKeyboardButton("‹ Trang trước", callback_data=f"menu:ws:{norm_page - 1}"))
        nav_row.append(InlineKeyboardButton(f"{norm_page + 1}/{total_pages}", callback_data="noop"))
        if norm_page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("Trang sau ›", callback_data=f"menu:ws:{norm_page + 1}"))
        buttons.append(nav_row)

    # Action row
    bottom_row: List[InlineKeyboardButton] = []
    if current_path.resolve() != base_path.resolve():
        bottom_row.append(InlineKeyboardButton("🔄 Về thư mục gốc", callback_data="set:ws:clear"))
    bottom_row.append(InlineKeyboardButton("‹ Quay lại Menu", callback_data="menu:main"))
    buttons.append(bottom_row)

    return InlineKeyboardMarkup(buttons)


def menu_keyboard() -> InlineKeyboardMarkup:
    """Build central navigation keyboard similar to AGY CLI."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🤖 Chọn Model", callback_data="menu:model"),
                InlineKeyboardButton("⚙️ Chế độ (/mode)", callback_data="menu:mode"),
            ],
            [
                InlineKeyboardButton("📁 Workspace", callback_data="menu:workspace"),
                InlineKeyboardButton("📊 Token & Chi phí", callback_data="action:tokens"),
            ],
            [
                InlineKeyboardButton("🗜️ Nén ngữ cảnh", callback_data="action:compact"),
                InlineKeyboardButton("🛑 Dừng tác vụ", callback_data="action:cancel"),
            ],
            [
                InlineKeyboardButton("🆕 Phiên mới (/new)", callback_data="action:new_session"),
                InlineKeyboardButton("📂 Danh sách Sessions", callback_data="action:sessions"),
            ],
        ]
    )


def tokens_keyboard() -> InlineKeyboardMarkup:
    """Build quick action keyboard for tokens/cost report."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🗜️ Nén ngữ cảnh (/compact)", callback_data="action:compact"),
                InlineKeyboardButton("🔄 Làm mới", callback_data="action:tokens"),
            ],
            [
                InlineKeyboardButton("‹ Quay lại Menu", callback_data="menu:main"),
            ],
        ]
    )
