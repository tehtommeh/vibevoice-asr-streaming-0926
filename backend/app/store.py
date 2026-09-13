"""Small JSON-backed store for modes, learned vocabulary and voice commands.

These live on the server rather than in browser storage so the desktop client
can inherit exactly what was tuned in the browser -- the same reasoning that
puts the OpenRouter call behind the backend.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

STORE_PATH = os.environ.get("STORE_PATH", "/data/store.json")
_lock = threading.Lock()

# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

_EDIT_RULES = (
    "Rules:\n"
    "- Return ONLY the edited passage. No preamble, no explanation, no quotes "
    "around it, no code fences.\n"
    "- Apply exactly what the instruction asks, and nothing more. Leave wording "
    "you were not asked to change alone.\n"
    "- The instruction reached you through speech recognition and may contain "
    "mis-heard words. Read it charitably.\n"
    "- If the instruction names a term that appears garbled in the passage, fix "
    "the passage to use the term as the instruction gives it.\n"
    "- If you cannot tell what is being asked, return the passage unchanged."
)

_DICTATE_RULES = (
    "You are cleaning up raw dictation. Return ONLY the cleaned text -- no "
    "preamble, no explanation, no quotes, no code fences.\n\n"
    "Rules:\n"
    "- Resolve spoken self-corrections. 'send it to Bob, no wait, Alice' becomes "
    "'send it to Alice'. Drop the retracted version entirely.\n"
    "- Remove filler: um, uh, like, you know, sort of, I mean (when used as "
    "filler rather than meaning).\n"
    "- Remove accidental repetition: 'the the', 'I I just'.\n"
    "- Fix punctuation, capitalisation and obvious speech-recognition slips.\n"
    "- Spell out nothing and normalise nothing else. Keep the speaker's words, "
    "register and vocabulary. Do not make it more formal, shorter or better.\n"
    "- If the dictation is already clean, return it unchanged."
)

DEFAULT_MODES = [
    {
        "id": "prose",
        "name": "Prose",
        "model": "",
        "temperature": 0.2,
        "system_prompt": (
            "You are a text editor driven by voice. You are given a passage and a "
            "spoken instruction describing how to change it.\n\n" + _EDIT_RULES +
            "\n- Preserve the author's voice, register and formatting conventions "
            "unless the instruction asks you to change them."
        ),
        "dictation_prompt": _DICTATE_RULES,
    },
    {
        "id": "email",
        "name": "Email",
        "model": "",
        "temperature": 0.2,
        "system_prompt": (
            "You are editing an email. You are given a passage and a spoken "
            "instruction describing how to change it.\n\n" + _EDIT_RULES +
            "\n- Keep it courteous and direct. Do not add greetings or sign-offs "
            "that were not asked for."
        ),
        "dictation_prompt": _DICTATE_RULES + "\n- This is email prose: keep sentences whole and punctuated.",
    },
    {
        "id": "commit",
        "name": "Commit message",
        "model": "",
        "temperature": 0.1,
        "system_prompt": (
            "You are editing a git commit message. You are given a passage and a "
            "spoken instruction describing how to change it.\n\n" + _EDIT_RULES +
            "\n- Subject line in the imperative mood, no trailing full stop, ideally "
            "under 72 characters. Body explains why, not what."
        ),
        "dictation_prompt": (
            _DICTATE_RULES +
            "\n- This is a commit message: imperative mood, no trailing full stop on "
            "the subject line."
        ),
    },
    {
        "id": "chat",
        "name": "Chat / Slack",
        "model": "",
        "temperature": 0.3,
        "system_prompt": (
            "You are editing a short chat message. You are given a passage and a "
            "spoken instruction describing how to change it.\n\n" + _EDIT_RULES +
            "\n- Keep it casual and brief. Do not pad it out or make it formal."
        ),
        "dictation_prompt": _DICTATE_RULES + "\n- This is a chat message: casual is correct, keep it short.",
    },
    {
        "id": "code",
        "name": "Code comment",
        "model": "",
        "temperature": 0.1,
        "system_prompt": (
            "You are editing a source-code comment or docstring. You are given a "
            "passage and a spoken instruction describing how to change it.\n\n" + _EDIT_RULES +
            "\n- Terse and technical. Preserve indentation and comment markers exactly."
        ),
        "dictation_prompt": (
            _DICTATE_RULES +
            "\n- This is a code comment: terse and technical. Identifiers stay in "
            "their written form (camelCase, snake_case) rather than being spelled out."
        ),
    },
]

# --------------------------------------------------------------------------
# Voice commands -- matched client-side, defined here so every client agrees.
# `action` is interpreted by the client; the browser and the desktop app do
# different things for the same word.
# --------------------------------------------------------------------------

DEFAULT_COMMANDS = [
    {"action": "cancel", "phrases": ["never mind", "nevermind", "scratch that", "cancel that",
                                     "forget it", "forget that", "cancel"]},
    {"action": "undo", "phrases": ["undo", "undo that", "revert that", "take that back"]},
    {"action": "redo", "phrases": ["redo", "redo that"]},
    {"action": "delete", "phrases": ["delete that", "remove that", "delete this", "clear that"]},
    {"action": "uppercase", "phrases": ["all caps", "uppercase that", "make it all caps",
                                        "capitalise that", "capitalize that"]},
    {"action": "lowercase", "phrases": ["lowercase that", "make it lowercase", "no caps"]},
    {"action": "titlecase", "phrases": ["title case", "title case that"]},
    {"action": "sentencecase", "phrases": ["sentence case", "sentence case that"]},
    {"action": "trim", "phrases": ["trim that", "strip whitespace"]},
    {"action": "newline", "phrases": ["new line", "newline"]},
    {"action": "paragraph", "phrases": ["new paragraph", "paragraph break"]},
]

# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _default_state() -> dict:
    return {"modes": DEFAULT_MODES, "vocab": {}, "active_mode": "prose"}


def _read() -> dict:
    if not os.path.isfile(STORE_PATH):
        return _default_state()
    try:
        with open(STORE_PATH) as fh:
            data = json.load(fh)
    except Exception:
        return _default_state()
    state = _default_state()
    if isinstance(data.get("modes"), list) and data["modes"]:
        state["modes"] = data["modes"]
    if isinstance(data.get("vocab"), dict):
        state["vocab"] = data["vocab"]
    if isinstance(data.get("active_mode"), str):
        state["active_mode"] = data["active_mode"]
    return state


def _write(state: dict) -> None:
    directory = os.path.dirname(STORE_PATH) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = STORE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, STORE_PATH)


def read_state() -> dict:
    with _lock:
        return _read()


def modes() -> list[dict]:
    return read_state()["modes"]


def get_mode(mode_id: Optional[str]) -> dict:
    state = read_state()
    wanted = mode_id or state.get("active_mode") or "prose"
    for mode in state["modes"]:
        if mode.get("id") == wanted:
            return mode
    return state["modes"][0]


def save_modes(new_modes: list[dict], active: Optional[str] = None) -> dict:
    with _lock:
        state = _read()
        state["modes"] = new_modes
        if active:
            state["active_mode"] = active
        _write(state)
        return state


def set_active_mode(mode_id: str) -> dict:
    with _lock:
        state = _read()
        state["active_mode"] = mode_id
        _write(state)
        return state


# --------------------------------------------------------------------------
# Learned vocabulary
# --------------------------------------------------------------------------

VOCAB_PROMOTE_AT = 2  # corrections seen this many times become ASR hotwords


def learn(term: str, heard: Optional[str] = None) -> dict:
    """Record that `term` is a word this user actually means."""
    term = (term or "").strip()
    if not term or len(term) > 80:
        return read_state()["vocab"]
    with _lock:
        state = _read()
        entry = state["vocab"].get(term) or {"count": 0, "heard": []}
        entry["count"] = int(entry.get("count", 0)) + 1
        if heard:
            heard = heard.strip()[:80]
            misheard = [h for h in entry.get("heard", []) if h.lower() != heard.lower()]
            entry["heard"] = ([heard] + misheard)[:5]
        state["vocab"][term] = entry
        _write(state)
        return state["vocab"]


def forget(term: str) -> dict:
    with _lock:
        state = _read()
        state["vocab"].pop(term, None)
        _write(state)
        return state["vocab"]


def hotwords(limit: int = 40) -> list[str]:
    """Terms seen often enough to be worth biasing the recogniser toward."""
    vocab = read_state()["vocab"]
    ranked = sorted(
        (t for t, e in vocab.items() if e.get("count", 0) >= VOCAB_PROMOTE_AT),
        key=lambda t: -vocab[t].get("count", 0),
    )
    return ranked[:limit]
