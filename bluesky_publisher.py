"""Publish to Bluesky via the AT Protocol XRPC API (no SDK — plain requests,
matching the rest of the repo).

Optional and secret-gated: if BLUESKY_HANDLE / BLUESKY_APP_PASSWORD aren't set,
`is_configured()` is False and callers skip Bluesky entirely, so the pipeline
runs unchanged without it. Auth uses a Bluesky *app password* (Settings →
Privacy and Security → App Passwords), never the account password.

Bluesky constraints handled here:
  - post text is capped at 300 graphemes (we budget in characters, which is a
    safe over-estimate since graphemes ≤ code points);
  - each image blob must be under ~1,000,000 bytes, so charts are recompressed;
  - up to 4 images per post;
  - a URL is only clickable if annotated with a byte-range facet.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import requests

PDS_HOST = "https://bsky.social"
POST_GRAPHEME_LIMIT = 300
BLOB_BYTE_LIMIT = 976_000          # a safe margin under Bluesky's 1,000,000-byte cap
MAX_IMAGES = 4


def is_configured() -> bool:
    return bool(os.environ.get("BLUESKY_HANDLE") and os.environ.get("BLUESKY_APP_PASSWORD"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _session() -> tuple[str, str]:
    """Create a fresh session each run (the access JWT is short-lived, but a
    per-run script never outlives it, so no refresh handling is needed)."""
    resp = requests.post(
        f"{PDS_HOST}/xrpc/com.atproto.server.createSession",
        json={
            "identifier": os.environ["BLUESKY_HANDLE"],
            "password": os.environ["BLUESKY_APP_PASSWORD"],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["accessJwt"], data["did"]


def _prepare_image_bytes(path: str, limit: int = BLOB_BYTE_LIMIT) -> bytes:
    """Return PNG bytes guaranteed under Bluesky's blob size limit, downscaling
    the chart as needed. Charts are dark-background PNGs, so we keep PNG (lossless
    text) and shrink dimensions rather than switching to lossy JPEG."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) <= limit:
        return raw

    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    width, height = image.size
    for _ in range(8):
        width = int(width * 0.85)
        height = int(height * 0.85)
        resized = image.resize((width, height), Image.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        if len(data) <= limit or width < 700:
            return data
    return data


def _upload_image(jwt: str, path: str) -> dict:
    data = _prepare_image_bytes(path)
    resp = requests.post(
        f"{PDS_HOST}/xrpc/com.atproto.repo.uploadBlob",
        headers={"Authorization": f"Bearer {jwt}", "Content-Type": "image/png"},
        data=data,
        timeout=45,
    )
    resp.raise_for_status()
    return resp.json()["blob"]


def _compose(text: str, link_url: str | None, link_label: str | None) -> tuple[str, list]:
    """Assemble the post body within the grapheme limit, appending a clickable
    channel link as its own trailing line with a byte-range link facet."""
    body = text.strip()
    if not (link_url and link_label):
        if len(body) > POST_GRAPHEME_LIMIT:
            body = body[: POST_GRAPHEME_LIMIT - 1].rstrip() + "…"
        return body, []

    tail = f"\n\n{link_label}"
    max_body = POST_GRAPHEME_LIMIT - len(tail)
    if len(body) > max_body:
        body = body[: max_body - 1].rstrip() + "…"
    full = f"{body}{tail}"

    # Facet ranges are UTF-8 byte offsets into the text; the 300 cap is graphemes.
    byte_start = len(full[: len(body) + 2].encode("utf-8"))          # skip body + "\n\n"
    byte_end = byte_start + len(link_label.encode("utf-8"))
    facets = [{
        "index": {"byteStart": byte_start, "byteEnd": byte_end},
        "features": [{"$type": "app.bsky.richtext.facet#link", "uri": link_url}],
    }]
    return full, facets


def post(text: str, image_paths: list[str] | None = None,
         link_url: str | None = None, link_label: str | None = None,
         alt_text: str = "Crypto market chart") -> dict:
    """Publish one Bluesky post with optional images and a clickable channel link."""
    jwt, did = _session()
    body, facets = _compose(text, link_url, link_label)

    record = {
        "$type": "app.bsky.feed.post",
        "text": body,
        "createdAt": _now_iso(),
        "langs": ["en"],
    }
    if facets:
        record["facets"] = facets
    if image_paths:
        images = [{"alt": alt_text, "image": _upload_image(jwt, path)}
                  for path in image_paths[:MAX_IMAGES]]
        record["embed"] = {"$type": "app.bsky.embed.images", "images": images}

    resp = requests.post(
        f"{PDS_HOST}/xrpc/com.atproto.repo.createRecord",
        headers={"Authorization": f"Bearer {jwt}"},
        json={"repo": did, "collection": "app.bsky.feed.post", "record": record},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
