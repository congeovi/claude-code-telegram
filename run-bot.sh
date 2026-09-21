#!/usr/bin/env bash
# Supervisor cho claude-code-telegram bot.
# Tự khởi động lại bot khi nó thoát (vd sau lệnh /restart trong Telegram).
# Dừng hẳn: tmux kill-session -t claude-code-tg  (hoặc Ctrl-C khi attach)
set -u
cd /home/cong/claude-code-telegram

while true; do
  echo "=== [supervisor] starting bot at $(date -u +%FT%TZ) ==="
  ./.venv/bin/claude-telegram-bot
  code=$?
  echo "=== [supervisor] bot exited code=$code, restart sau 2s ==="
  sleep 2
done
