"""Input guardrails: prompt injection detection, PII redaction, toxicity filtering.

These checks run BEFORE the query enters the agent graph so that no harmful,
malicious, or privacy-violating content reaches the LLM or retrieval engine.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GuardrailResult:
    """Result returned by any guardrail check."""

    allowed: bool
    """True if the query/response passed the guardrail."""

    modified_text: str
    """The (potentially redacted/modified) text to use downstream."""

    violations: list[str] = field(default_factory=list)
    """List of human-readable violation descriptions."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional diagnostic information (matched patterns, entity types, etc.)."""


# ---------------------------------------------------------------------------
# Prompt Injection Guard
# ---------------------------------------------------------------------------

# English and Arabic prompt injection patterns
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions?", "prompt_injection_ignore_instructions"),
    (r"disregard\s+(all\s+)?prior\s+(instructions?|context)", "prompt_injection_disregard"),
    (r"system\s*prompt\s*override", "prompt_injection_override"),
    (r"bypass\s+safety\s+filter", "prompt_injection_bypass"),
    (r"you\s+are\s+now\s+in\s+developer\s+mode", "prompt_injection_dev_mode"),
    (r"act\s+as\s+(if\s+you\s+are\s+)?an?\s+(unrestricted|uncensored|evil|jailbroken)", "prompt_injection_persona"),
    (r"jailbreak", "prompt_injection_jailbreak"),
    (r"dan\s+(mode|prompt)", "prompt_injection_dan"),
    (r"<\s*script[^>]*>", "xss_script_injection"),
    (r"(--|;)\s*(drop|delete|truncate|insert|update)\s+\w+", "sql_injection"),
    # Arabic patterns
    (r"تجاهل\s+(كل\s+)?التعليمات\s+السابقة", "prompt_injection_ar_ignore"),
    (r"تخطى\s+إجراءات\s+الأمان", "prompt_injection_ar_bypass"),
    (r"أنت\s+الآن\s+في\s+وضع\s+المطور", "prompt_injection_ar_dev_mode"),
    (r"تجاوز\s+القيود", "prompt_injection_ar_bypass_restrictions"),
    (r"وضع\s+غير\s+مقيد", "prompt_injection_ar_unrestricted"),
]


def check_prompt_injection(text: str) -> GuardrailResult:
    """Detect prompt injection attempts in English and Arabic.

    Uses a set of regex patterns covering common jailbreak, persona override,
    SQL/XSS injection, and Arabic-language injection attempts.

    Args:
        text: The raw user query.

    Returns:
        GuardrailResult with allowed=False if injection is detected.
    """
    violations: list[str] = []
    matched_patterns: list[str] = []

    for pattern, tag in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            violations.append(f"Prompt injection pattern detected: {tag}")
            matched_patterns.append(tag)
            logger.warning("Prompt injection detected: pattern=%s, hash=%s", tag, hashlib.md5(text.encode()).hexdigest()[:8])

    return GuardrailResult(
        allowed=len(violations) == 0,
        modified_text=text,
        violations=violations,
        metadata={"matched_patterns": matched_patterns},
    )


# ---------------------------------------------------------------------------
# PII Redaction Guard
# ---------------------------------------------------------------------------

# PII entity patterns (English + Arabic)
_PII_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, replacement_tag, entity_type)
    (
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        "EMAIL",
    ),
    (
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[PHONE]",
        "PHONE_US",
    ),
    (
        r"\b(?:\+?9665|05)\d{8}\b",
        "[PHONE]",
        "PHONE_SA",
    ),
    (
        r"\b(?:\+?974|[3-7])\d{7}\b",
        "[PHONE]",
        "PHONE_QA",
    ),
    (
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "[SSN]",
        "SSN",
    ),
    (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        "[CREDIT_CARD]",
        "CREDIT_CARD",
    ),
    (
        r"\b(?:https?://|www\.)\S+\b",
        "[URL]",
        "URL",
    ),
    (
        r"\b\d{1,5}\s+[\w\s]{3,30},?\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|شارع|طريق|حي|شقة)\b",
        "[ADDRESS]",
        "ADDRESS",
    ),
    (
        r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        "[NAME]",
        "PERSON_NAME_EN",
    ),
    (
        r"\b(?:QID|رقم\s*الهوية|هوية|جواز\s*سفر|ID|SSN)\s*:?\s*\d{5,15}\b",
        "[ID_NUMBER]",
        "QATARI_ID",
    ),
]


def redact_pii(text: str, *, strict: bool = False) -> GuardrailResult:
    """Detect and redact PII from user queries.

    In strict mode, returns allowed=False if PII is found (for high-security
    environments). In normal mode, redacts PII and marks allowed=True so the
    sanitized query can proceed.

    Args:
        text: The raw user query.
        strict: If True, block queries containing PII rather than redacting.

    Returns:
        GuardrailResult with modified_text having PII replaced by tags.
    """
    modified = text
    found_entities: list[str] = []
    violations: list[str] = []

    for pattern, replacement, entity_type in _PII_PATTERNS:
        matches = re.findall(pattern, modified, re.IGNORECASE | re.UNICODE)
        if matches:
            found_entities.append(f"{entity_type} ({len(matches)} instance(s))")
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE | re.UNICODE)
            if strict:
                violations.append(f"PII detected and query blocked: {entity_type}")
            logger.info("PII redacted: type=%s, count=%d", entity_type, len(matches))

    return GuardrailResult(
        allowed=not (strict and bool(found_entities)),
        modified_text=modified,
        violations=violations,
        metadata={"pii_entities_found": found_entities, "pii_redacted": bool(found_entities)},
    )


# ---------------------------------------------------------------------------
# Toxicity / Harmful Content Filter
# ---------------------------------------------------------------------------

# Heuristic keyword lists — in production replace with a toxicity classifier
_TOXIC_PATTERNS: list[tuple[str, str]] = [
    (r"\b(how\s+to\s+(make|build|create|synthesize)\s+(a\s+|an\s+)?(bomb|explosive|poison|weapon))\b", "weapons_creation"),
    (r"\b(suicide\s+methods?|how\s+to\s+kill\s+myself|self[-\s]harm\s+instructions?)\b", "self_harm"),
    (r"\b(child\s+(pornography|exploitation|abuse\s+material))\b", "csam"),
    (r"\b(step[- ]by[- ]step\s+guide\s+to\s+hack(ing)?)\b", "hacking_guide"),
    (r"\b(make\s+methamphetamine|cook\s+meth|drug\s+synthesis)\b", "drug_synthesis"),
    # Arabic harmful patterns
    (r"(كيف\s+أصنع\s+قنبلة|تعليمات\s+صنع\s+متفجرات)", "weapons_ar"),
    (r"(كيف\s+أؤذي\s+نفسي|طرق\s+الانتحار)", "self_harm_ar"),
]


def check_toxicity(text: str) -> GuardrailResult:
    """Heuristic toxicity detection for harmful content requests.

    In production this should be supplemented with a dedicated toxicity
    classification model (e.g., Detoxify, Perspective API).

    Args:
        text: The raw user query.

    Returns:
        GuardrailResult with allowed=False if harmful content is detected.
    """
    violations: list[str] = []
    matched: list[str] = []

    for pattern, tag in _TOXIC_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            violations.append(f"Potentially harmful content detected: {tag}")
            matched.append(tag)
            logger.warning("Toxic content detected: tag=%s", tag)

    return GuardrailResult(
        allowed=len(violations) == 0,
        modified_text=text,
        violations=violations,
        metadata={"toxic_categories": matched},
    )


# ---------------------------------------------------------------------------
# Query Length and Complexity Limits
# ---------------------------------------------------------------------------

MAX_QUERY_CHARS = 3000
MIN_QUERY_CHARS = 2
MAX_QUERY_NEWLINES = 30


def check_query_limits(text: str) -> GuardrailResult:
    """Enforce query length and structural sanity limits.

    Prevents abuse via excessively long prompts, empty queries, or
    multi-thousand-line structured payloads.

    Args:
        text: The raw user query.

    Returns:
        GuardrailResult with allowed=False if limits are exceeded.
    """
    violations: list[str] = []

    if len(text) < MIN_QUERY_CHARS:
        violations.append(f"Query too short (min {MIN_QUERY_CHARS} characters).")
    if len(text) > MAX_QUERY_CHARS:
        violations.append(f"Query exceeds maximum length ({MAX_QUERY_CHARS} characters).")
    if text.count("\n") > MAX_QUERY_NEWLINES:
        violations.append(f"Query contains too many newlines (max {MAX_QUERY_NEWLINES}).")

    return GuardrailResult(
        allowed=len(violations) == 0,
        modified_text=text[:MAX_QUERY_CHARS] if len(text) > MAX_QUERY_CHARS else text,
        violations=violations,
        metadata={"query_length": len(text), "newline_count": text.count("\n")},
    )


# ---------------------------------------------------------------------------
# Composite Input Guard — runs all checks in sequence
# ---------------------------------------------------------------------------


def run_input_guards(text: str, *, pii_strict: bool = False) -> GuardrailResult:
    """Run all input guardrails in sequence and return a consolidated result.

    Pipeline:
        1. Query limit checks (length, newlines)
        2. Prompt injection detection
        3. Toxicity / harmful content detection
        4. PII detection and redaction

    Args:
        text: The raw user query.
        pii_strict: If True, block queries with PII instead of redacting them.

    Returns:
        GuardrailResult. If allowed=False, the caller should reject the request.
        If allowed=True, use modified_text as the sanitized query.
    """
    all_violations: list[str] = []
    combined_metadata: dict[str, Any] = {}

    # Step 1: Length limits
    limit_result = check_query_limits(text)
    all_violations.extend(limit_result.violations)
    combined_metadata.update(limit_result.metadata)
    if not limit_result.allowed:
        return GuardrailResult(
            allowed=False,
            modified_text=limit_result.modified_text,
            violations=all_violations,
            metadata=combined_metadata,
        )

    # Step 2: Prompt injection
    injection_result = check_prompt_injection(text)
    all_violations.extend(injection_result.violations)
    combined_metadata.update(injection_result.metadata)
    if not injection_result.allowed:
        return GuardrailResult(
            allowed=False,
            modified_text=text,
            violations=all_violations,
            metadata=combined_metadata,
        )

    # Step 3: Toxicity
    toxicity_result = check_toxicity(text)
    all_violations.extend(toxicity_result.violations)
    combined_metadata.update(toxicity_result.metadata)
    if not toxicity_result.allowed:
        return GuardrailResult(
            allowed=False,
            modified_text=text,
            violations=all_violations,
            metadata=combined_metadata,
        )

    # Step 4: PII (redact, do not block unless strict mode)
    pii_result = redact_pii(text, strict=pii_strict)
    all_violations.extend(pii_result.violations)
    combined_metadata.update(pii_result.metadata)

    return GuardrailResult(
        allowed=pii_result.allowed,
        modified_text=pii_result.modified_text,
        violations=all_violations,
        metadata=combined_metadata,
    )
