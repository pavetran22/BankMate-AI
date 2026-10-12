"""Input and output guardrails.

Input (before the model sees anything):
  * length limit
  * PII masking: card numbers (keep last 4) and MyKad numbers
  * credential detection: a customer typing their OTP / PIN / password gets it
    masked and a safety warning, because the bank never needs those
  * prompt-injection heuristics: attempts to override instructions, reveal the
    system prompt, or reach other customers' data are refused

Output (before the user sees the answer):
  * PII masking again, in case the model echoes something sensitive

These are deliberately simple, deterministic rules: cheap, testable and
auditable. In production you would layer a model-based classifier on top.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings

CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
MYKAD_RE = re.compile(r"\b\d{6}-?\d{2}-?\d{4}\b")
CREDENTIAL_RE = re.compile(
    r"\b(otp|tac|pin|password|passcode|pwd)\b\s*(?:is|:|=|-)?\s*((?=[A-Za-z@#$%^&*!]*\d)[A-Za-z0-9@#$%^&*!]{4,})",
    re.IGNORECASE,
)
INJECTION_PATTERNS = [
    (r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
     "attempt to override instructions"),
    (r"disregard (your|the|all) (rules|instructions|guidelines)", "attempt to override instructions"),
    (r"(reveal|show|print|repeat|leak).{0,30}(system prompt|instructions|hidden prompt)",
     "attempt to extract the system prompt"),
    (r"\byou are now\b|\bact as (an? )?(admin|developer|dan)\b|\bjailbreak\b|developer mode",
     "attempt to change the assistant's role"),
    (r"\b(other|another|all) customers?'?s?\b.{0,40}\b(data|transactions|accounts?|details)\b",
     "request for other customers' data"),
    # Bare customer ids like "C004" (but not transaction ids like "TX-C002-9001")
    (r"(?<![\w-])C\d{3}(?![\w-])", "reference to a customer id"),
    (r"\b(drop|delete|update|insert)\b.{0,20}\b(table|from|into)\b|;\s*--", "possible SQL injection"),
]

REFUSAL = (
    "Sorry, I can't help with that request. I can answer questions about Nusa Bank "
    "products and policies, and help you review activity on your own account."
)
CREDENTIAL_WARNING = (
    "Security reminder: never share your OTP, PIN or password with anyone, including "
    "bank staff or this assistant. Nusa Bank will never ask for them. I've hidden "
    "what you typed. If you shared it anywhere else, use the Kill Switch in the app "
    "or call the fraud hotline at 1-300-88-6872 now."
)


@dataclass
class GuardrailResult:
    allowed: bool
    text: str                                   # sanitised text to pass on
    flags: list[str] = field(default_factory=list)
    message: str | None = None                  # refusal or warning for the user


def mask_pii(text: str) -> tuple[str, list[str]]:
    flags = []

    def _card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19:
            flags.append("card_number_masked")
            return f"**** **** **** {digits[-4:]}"
        return m.group()

    text = CARD_RE.sub(_card, text)
    if MYKAD_RE.search(text):
        flags.append("mykad_masked")
        text = MYKAD_RE.sub("******-**-****", text)
    return text, flags


def check_input(text: str) -> GuardrailResult:
    text = (text or "").strip()
    if not text:
        return GuardrailResult(False, "", ["empty"], "Please type a question.")
    if len(text) > settings.max_input_chars:
        return GuardrailResult(False, "", ["too_long"],
                               f"Please keep your message under {settings.max_input_chars} characters.")

    text, flags = mask_pii(text)

    message = None
    if CREDENTIAL_RE.search(text):
        text = CREDENTIAL_RE.sub(lambda m: f"{m.group(1)} [REDACTED]", text)
        flags.append("credential_redacted")
        message = CREDENTIAL_WARNING

    for pattern, reason in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(False, text, flags + [f"blocked: {reason}"], REFUSAL)

    return GuardrailResult(True, text, flags, message)


def check_output(text: str) -> tuple[str, list[str]]:
    return mask_pii(text or "")
