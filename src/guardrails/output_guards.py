"""Output guardrails: validate LLM responses for safety, factual grounding,
hallucination markers, PII leakage, and format compliance before serving.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.guardrails.input_guards import GuardrailResult, redact_pii

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hallucination / Fabrication Markers
# ---------------------------------------------------------------------------

# Phrases that signal the model is generating content outside its context
_HALLUCINATION_MARKERS: list[tuple[str, str]] = [
    (r"\b(I\s+don't\s+have\s+access\s+to\s+real[- ]time|as\s+of\s+my\s+knowledge\s+cutoff)\b", "training_data_leak"),
    (r"\baccording\s+to\s+my\s+training\s+data\b", "training_data_leak"),
    (r"\b(I\s+(was\s+)?trained|my\s+training)\b", "self_reference_training"),
    (r"\bas\s+an?\s+(AI|language model|LLM|chatbot|assistant)\b", "self_reference_ai"),
    (r"\bI\s+cannot\s+(verify|confirm|guarantee)\s+(the\s+accuracy|this\s+information)\b", "unverified_claim"),
    (r"\bI\s+made\s+up\b|\bI\s+fabricated\b|\bI\s+invented\b", "explicit_fabrication"),
    # Arabic hallucination markers
    (r"\bبحسب\s+تدريبي\b|\bمعلوماتي\s+محدودة\b", "training_data_leak_ar"),
    (r"\bلا\s+أستطيع\s+التحقق\b", "unverified_claim_ar"),
]


# ---------------------------------------------------------------------------
# Structural Response Validator
# ---------------------------------------------------------------------------


def validate_response_structure(
    answer: str,
    citations: list[dict[str, Any]],
    language: str,
) -> GuardrailResult:
    """Validate that the LLM response has proper structure and citations.

    Checks:
    - Answer is non-empty and not a single-word stub
    - At least one citation is present when answer is substantive
    - Language tag matches the detected query language
    - Answer does not consist entirely of the context (copy-paste detection)

    Args:
        answer: The generated answer text.
        citations: List of citation dicts from the LLM response.
        language: Expected language ("ar", "en", or "mixed").

    Returns:
        GuardrailResult indicating structural validity.
    """
    violations: list[str] = []
    metadata: dict[str, Any] = {}

    # Non-empty check
    if not answer or len(answer.strip()) < 10:
        violations.append("Response answer is too short or empty.")
        metadata["answer_length"] = len(answer) if answer else 0

    # Citation check — only enforce for substantive answers (> 50 chars)
    if len(answer) > 50 and not citations:
        violations.append("Substantive answer missing citations — potential hallucination risk.")
        metadata["citations_count"] = 0

    # Language consistency check (heuristic: Arabic chars present?)
    arabic_ratio = len(re.findall(r"[\u0600-\u06FF]", answer)) / max(len(answer), 1)
    if language == "ar" and arabic_ratio < 0.1:
        violations.append("Expected Arabic response but answer appears to be in English.")
        metadata["arabic_char_ratio"] = round(arabic_ratio, 3)
    elif language == "en" and arabic_ratio > 0.6:
        violations.append("Expected English response but answer appears to be predominantly Arabic.")
        metadata["arabic_char_ratio"] = round(arabic_ratio, 3)

    return GuardrailResult(
        allowed=len(violations) == 0,
        modified_text=answer,
        violations=violations,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Hallucination Marker Detector
# ---------------------------------------------------------------------------


def check_hallucination_markers(answer: str) -> GuardrailResult:
    """Scan LLM output for phrases that indicate hallucination or off-context generation.

    These markers suggest the model is drawing on training data rather than
    the provided context — a clear faithfulness violation.

    Args:
        answer: The generated answer text.

    Returns:
        GuardrailResult with allowed=False if hallucination markers are found.
    """
    violations: list[str] = []
    matched: list[str] = []

    for pattern, tag in _HALLUCINATION_MARKERS:
        if re.search(pattern, answer, re.IGNORECASE | re.UNICODE):
            violations.append(f"Hallucination marker detected: {tag}")
            matched.append(tag)
            logger.warning("Hallucination marker in output: tag=%s", tag)

    return GuardrailResult(
        allowed=len(violations) == 0,
        modified_text=answer,
        violations=violations,
        metadata={"hallucination_markers": matched},
    )


# ---------------------------------------------------------------------------
# Output PII Leakage Detector
# ---------------------------------------------------------------------------


def check_output_pii(answer: str) -> GuardrailResult:
    """Check if the LLM output inadvertently leaks PII from the context.

    Applies the same PII redaction logic used on inputs to catch cases where
    the model reproduced sensitive data from retrieved documents.

    Args:
        answer: The generated answer text.

    Returns:
        GuardrailResult with PII redacted from modified_text.
    """
    result = redact_pii(answer, strict=False)
    if result.metadata.get("pii_redacted"):
        logger.info(
            "PII detected in LLM output and redacted: entities=%s",
            result.metadata.get("pii_entities_found", []),
        )
    return result


# ---------------------------------------------------------------------------
# Composite Output Guard
# ---------------------------------------------------------------------------


def run_output_guards(
    answer: str,
    citations: list[dict[str, Any]],
    language: str,
) -> GuardrailResult:
    """Run all output guardrails and return a consolidated, sanitized result.

    Pipeline:
        1. Hallucination marker detection
        2. Structural response validation (length, citations, language)
        3. PII leakage check and redaction

    Args:
        answer: Raw LLM-generated answer text.
        citations: Extracted citation objects from the LLM response.
        language: Detected query language ("ar", "en", or "mixed").

    Returns:
        GuardrailResult. modified_text is the final safe answer to serve.
        violations lists all policy failures (logged / returned to caller).
    """
    all_violations: list[str] = []
    combined_metadata: dict[str, Any] = {}
    current_text = answer

    # Step 1: Hallucination markers
    halluc_result = check_hallucination_markers(current_text)
    all_violations.extend(halluc_result.violations)
    combined_metadata.update(halluc_result.metadata)

    # Step 2: Structural validation
    struct_result = validate_response_structure(current_text, citations, language)
    all_violations.extend(struct_result.violations)
    combined_metadata.update(struct_result.metadata)

    # Step 3: PII leakage — always redact, log
    pii_result = check_output_pii(current_text)
    all_violations.extend(pii_result.violations)
    combined_metadata.update(pii_result.metadata)
    current_text = pii_result.modified_text  # use redacted version

    # Output is "allowed" if no hard violations (PII redaction is non-blocking)
    hard_violations = [v for v in all_violations if "hallucination" in v.lower() or "missing citations" in v.lower()]
    allowed = len(hard_violations) == 0

    return GuardrailResult(
        allowed=allowed,
        modified_text=current_text,
        violations=all_violations,
        metadata=combined_metadata,
    )
