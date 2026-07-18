"""
Posts to Telegram using the Bot API directly (no extra SDK needed for
something this simple — matches the lightweight style of the rest of the repo).
"""

import os
import json
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


def post_charts(chat_id: str, image_paths: list[str], caption: str) -> dict:
    """Publish one image or a two-chart Telegram album as one content unit."""
    if len(image_paths) == 1:
        return post_chart(chat_id, image_paths[0], caption)

    caption = caption[: config.TELEGRAM_CAPTION_LIMIT]
    media = []
    files = {}
    try:
        for index, image_path in enumerate(image_paths):
            key = f"chart{index}"
            item = {"type": "photo", "media": f"attach://{key}"}
            if index == 0:
                item["caption"] = caption
            media.append(item)
            files[key] = open(image_path, "rb")
        response = requests.post(
            f"{_base_url()}/sendMediaGroup",
            data={"chat_id": chat_id, "media": json.dumps(media)},
            files=files,
            timeout=45,
        )
        response.raise_for_status()
        messages = response.json()["result"]
        result = dict(messages[0])
        result["album_message_ids"] = [message["message_id"] for message in messages]
        return result
    finally:
        for file_handle in files.values():
            file_handle.close()


def reply_with_photo(chat_id: str, reply_to_message_id: int, image_path: str, caption: str) -> dict:
    """Posts a chart threaded as a reply (used for the trade-signal open post)."""
    caption = caption[: config.TELEGRAM_CAPTION_LIMIT]
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{_base_url()}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption, "reply_to_message_id": reply_to_message_id},
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
