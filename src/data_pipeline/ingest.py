import logging
import os
import re
from typing import Any

from src.agents.nodes import detect_language_rule_based, load_fasttext_model
from src.config import get_settings

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text_content = []
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_content.append(t)
    except Exception as e:
        logger.error("Error reading PDF %s: %s", file_path, e)
    return "\n".join(text_content)


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    text_content = []
    try:
        import docx

        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text_content.append(para.text.strip())
    except Exception as e:
        logger.error("Error reading DOCX %s: %s", file_path, e)
    return "\n".join(text_content)


def extract_text_from_txt_or_md(file_path: str) -> str:
    """Extract text from plain text or Markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error("Error reading text file %s: %s", file_path, e)
        return ""


def detect_document_language(text: str) -> str:
    """Detect language of document content using FastText or rule-based heuristic."""
    settings = get_settings()
    ft_model = load_fasttext_model(settings.FASTTEXT_MODEL_PATH)

    if ft_model is not None:
        try:
            sample = text[:1000].replace("\n", " ")
            preds = ft_model.predict(sample, k=1)
            lang = preds[0][0].replace("__label__", "")
            if lang in ("ar", "en"):
                return lang
        except Exception:
            pass

    lang, _ = detect_language_rule_based(text[:1000])
    return lang if lang in ("ar", "en") else "en"


def ingest_documents(source_dir: str) -> list[dict[str, Any]]:
    """Ingest PDF, DOCX, TXT, and MD documents from source directory."""
    if not os.path.exists(source_dir):
        logger.warning("Source directory does not exist: %s", source_dir)
        return []

    documents: list[dict[str, Any]] = []

    for root, _, files in os.walk(source_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".pdf", ".docx", ".txt", ".md"]:
                continue

            file_path = os.path.join(root, filename)
            doc_id = os.path.splitext(filename)[0]
            # Sanitize doc_id
            doc_id = re.sub(r"[^\w\-]", "_", doc_id)

            text = ""
            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif ext == ".docx":
                text = extract_text_from_docx(file_path)
            elif ext in [".txt", ".md"]:
                text = extract_text_from_txt_or_md(file_path)

            if not text.strip():
                logger.warning("Skipping empty document: %s", file_path)
                continue

            lang = detect_document_language(text)
            title = os.path.splitext(filename)[0].replace("_", " ").title()

            doc_record = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "language": lang,
                "source_path": file_path,
                "metadata": {
                    "filename": filename,
                    "extension": ext,
                    "source": filename,
                    "title": title,
                    "length": len(text),
                },
            }
            documents.append(doc_record)

    logger.info("Ingested %d documents from %s", len(documents), source_dir)
    return documents
