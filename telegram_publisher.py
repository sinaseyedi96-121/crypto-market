"""
Posts to Telegram using the Bot API directly (no extra SDK needed for
something this simple — matches the lightweight style of the rest of the repo).
"""

import os
import requests
import config


def _base_url() -> str:
    token = os.environ["TELEGRAM_TOKEN"]
    return f"https://api.telegram.org/bot{token}"


def post_chart(chat_id: str, image_path: str, caption: str) -> dict:
    """Posts a chart with caption. Returns the Telegram message object
    (includes message_id, which state_manager stores for later follow-up replies)."""
    caption = caption[: config.TELEGRAM_CAPTION_LIMIT]
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{_base_url()}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": f},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["result"]


def reply_to_message(chat_id: str, reply_to_message_id: int, text: str) -> dict:
    """Posts a text reply threaded under a previous post (used for follow-ups)."""
    resp = requests.post(
        f"{_base_url()}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": text,
            "reply_to_message_id": reply_to_message_id,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["result"]
