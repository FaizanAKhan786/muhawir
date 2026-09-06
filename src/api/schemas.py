import re
from typing import Any
from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question in Arabic or English")
    session_id: str | None = Field(default=None, description="Optional session or conversation ID")

    @model_validator(mode="after")
    def validate_and_sanitize_query(self) -> "ChatRequest":
        q = self.query.strip()
        # Prompt injection patterns
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"disregard\s+all\s+prior",
            r"system\s+prompt\s+override",
            r"bypass\s+safety\s+filters",
            r"you\s+are\s+now\s+in\s+developer\s+mode",
            r"jailbreak",
            r"تجاهل\s+(كل\s+)?التعليمات\s+السابقة",
            r"تخطى\s+إجراءات\s+الأمان",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, q, re.IGNORECASE):
                raise ValueError("Potentially malicious prompt injection pattern detected in query.")
        self.query = q
        return self


class CitationModel(BaseModel):
    doc_id: str
    chunk_id: str
    text_snippet: str
    source: str


class QualityScoreModel(BaseModel):
    faithfulness: float
    language_consistent: bool
    grounded: bool


class ChatResponse(BaseModel):
    answer: str
    language: str
    citations: list[CitationModel] = []
    quality_score: QualityScoreModel | None = None
    confidence: float = 1.0
    trace_id: str | None = None


class EvalRequest(BaseModel):
    golden_set: str = Field(default="data/golden_sets/golden_v1.0.json", description="Path to evaluation dataset")
    log_to_mlflow: bool = Field(default=True, description="Whether to record metrics into MLflow")


class EvalResponse(BaseModel):
    metrics: dict[str, Any]
    passed: bool
    failures: list[str] = []


class ServiceStatus(BaseModel):
    status: str
    latency_ms: float | None = None
    details: str | None = None


class HealthResponse(BaseModel):
    status: str
    services: dict[str, ServiceStatus]
