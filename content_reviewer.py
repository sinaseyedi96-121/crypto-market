"""Weekly OpenAI critique of recent Telegram posts and the hypothetical
signal track record. Reads posts_log.jsonl, asks OpenAI for a structured
review, and — only when it has a concrete, justified change — writes updated
versions of narrative_generator.py and/or config.py for the calling workflow
to test and open as a pull request. Never commits or pushes anything itself.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from openai import OpenAI

import config


REVIEW_SYSTEM_PROMPT = """You are a careful reviewer of an automated crypto-education Telegram channel.

You will be given: recent post captions, the hypothetical-signal track record (win rate, average R), and the full current source of the two files you are allowed to change.

Write your review as markdown with three parts: Scientific accuracy (are technical claims consistent with the indicators actually used, is language properly hedged, nothing invented), Audience and marketing quality (clarity, repetitiveness across posts, hook/headline strength, whether it reads as generic or genuinely useful), and Hypothetical-signal calibration (is the win rate and average R reasonable, do target/stop placements look too tight or too loose, does every signal post clearly read as educational and hypothetical rather than advice).

Only propose file changes when you have a specific, justified improvement backed by something you observed in the posts or track record. If nothing concrete needs to change, propose none.

When you do propose a change, change only prompt wording (the string constants) or tunable numeric/string constants. Never change function names, signatures, imports, or control flow. Reproduce the entire file exactly as given except for your specific edit — do not truncate or summarize any part of it. Never remove or weaken the hypothetical/educational disclaimer wording, and never introduce wording that instructs the reader to enter, exit, or size a trade.

For each file you want to change, append a block in exactly this form, with the complete new file contents inside the fence:

<<<FILE filename.py>>>
```python
...full file contents...
```
<<<END FILE>>>

Put your markdown review first, then zero or more of these blocks after it. Do not use this block format for anything else.
"""


def _load_recent_posts(days: int) -> list[dict]:
    if not os.path.exists(config.POSTS_LOG_FILE):
        return []
    cutoff = time.time() - days * 86400
    posts = []
    with open(config.POSTS_LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("timestamp", 0) >= cutoff:
                posts.append(entry)
    return posts


def _signal_stats(posts: list[dict]) -> dict:
    closed = [p for p in posts if p.get("mode") == "signal_close"]
    wins = [p for p in closed if p.get("outcome") == "target"]
    avg_r = sum(p.get("r_multiple", 0) for p in closed) / len(closed) if closed else 0.0
    return {
        "closed_count": len(closed),
        "win_count": len(wins),
        "win_rate_pct": (len(wins) / len(closed) * 100) if closed else None,
        "avg_r_multiple": avg_r,
    }


def _format_posts_for_prompt(posts: list[dict]) -> str:
    lines = []
    for post in posts:
        when = time.strftime("%Y-%m-%d", time.gmtime(post.get("timestamp", 0)))
        label = post.get("ticker") or "/".join(post.get("assets", [])) or "market"
        caption = (post.get("caption") or "").replace("\n", " ")[:300]
        lines.append(f"[{when}] {post.get('mode')} {label}: {caption}")
    return "\n".join(lines) if lines else "No posts in this window."


def _read_source(filenames: list[str]) -> dict[str, str]:
    source = {}
    for filename in filenames:
        with open(filename, "r") as f:
            source[filename] = f.read()
    return source


def _build_context(posts: list[dict], stats: dict, source: dict[str, str]) -> str:
    parts = [
        f"Track record over the last {config.REVIEW_LOOKBACK_DAYS} days:",
        f"Closed hypothetical signals: {stats['closed_count']}",
        f"Wins: {stats['win_count']}",
        f"Win rate: {stats['win_rate_pct']:.1f}%" if stats["win_rate_pct"] is not None else "Win rate: n/a",
        f"Average R multiple: {stats['avg_r_multiple']:.2f}",
        "",
        "Recent posts:",
        _format_posts_for_prompt(posts),
        "",
    ]
    for filename, content in source.items():
        parts.append(f"--- current contents of {filename} ---")
        parts.append(content)
        parts.append(f"--- end of {filename} ---")
    return "\n".join(parts)


_FILE_BLOCK_RE = re.compile(
    r"<<<FILE\s+(\S+)>>>\s*```(?:\w+)?\n(.*?)```\s*<<<END FILE>>>",
    re.DOTALL,
)


def _extract_review(response_text: str) -> tuple[str, dict[str, str]]:
    first_marker = response_text.find("<<<FILE")
    review_markdown = response_text if first_marker == -1 else response_text[:first_marker].strip()

    proposed = {}
    for filename, content in _FILE_BLOCK_RE.findall(response_text):
        filename = filename.strip()
        if filename not in config.REVIEWER_EDITABLE_FILES:
            review_markdown += f"\n\n_Ignored a proposed change to `{filename}`: not an editable file._"
            continue
        proposed[filename] = content
    return review_markdown, proposed


def _call_openai(context: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        max_completion_tokens=config.REVIEWER_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def _syntax_check(filename: str) -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", filename],
        capture_output=True, text=True,
    )
    return None if result.returncode == 0 else result.stderr.strip()


def _run_tests() -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        capture_output=True, text=True,
    )
    return None if result.returncode == 0 else (result.stdout + result.stderr).strip()


def _revert(filenames: list[str]) -> None:
    if filenames:
        subprocess.run(["git", "checkout", "--", *filenames])


def apply_proposed_changes(proposed: dict[str, str]) -> tuple[list[str], str]:
    """Writes proposed files, then requires both a syntax check and the full
    test suite to pass before leaving any of them changed. Returns the list of
    filenames actually left modified, plus any rejection notes."""
    if not proposed:
        return [], ""

    notes = []
    written = []
    for filename, content in proposed.items():
        with open(filename, "w") as f:
            f.write(content)
        error = _syntax_check(filename)
        if error:
            notes.append(f"_Proposed change to `{filename}` rejected: syntax error — {error}_")
            _revert([filename])
        else:
            written.append(filename)

    if not written:
        return [], "\n\n".join(notes)

    test_error = _run_tests()
    if test_error:
        notes.append(f"_Proposed changes to {', '.join(written)} rejected: test suite failed._\n```\n{test_error[-2000:]}\n```")
        _revert(written)
        return [], "\n\n".join(notes)

    return written, "\n\n".join(notes)


def main() -> None:
    posts = _load_recent_posts(config.REVIEW_LOOKBACK_DAYS)
    stats = _signal_stats(posts)

    if not posts:
        report = "## Content review\n\nNo posts logged in the review window yet — nothing to critique."
        with open(config.REVIEW_REPORT_FILE, "w") as f:
            f.write(report)
        _set_output("changed", "false")
        print(report)
        return

    source = _read_source(config.REVIEWER_EDITABLE_FILES)
    context = _build_context(posts, stats, source)
    response = _call_openai(context)
    review_markdown, proposed = _extract_review(response)

    written, notes = apply_proposed_changes(proposed)

    report_parts = [f"## Content review ({len(posts)} posts, last {config.REVIEW_LOOKBACK_DAYS} days)", "", review_markdown]
    if notes:
        report_parts += ["", notes]
    if written:
        report_parts += ["", f"Applied and test-passed changes to: {', '.join(written)}."]
    report = "\n".join(report_parts)

    with open(config.REVIEW_REPORT_FILE, "w") as f:
        f.write(report)

    print(report)
    _set_output("changed", "true" if written else "false")


def _set_output(key: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
