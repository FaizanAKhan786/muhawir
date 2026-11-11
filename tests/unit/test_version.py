"""Unit tests for version metadata and system capabilities."""

from src import __version__, get_version_info, SYSTEM_CAPABILITIES


def test_get_version_info():
    """Verify version info returns correct metadata and capabilities."""
    info = get_version_info()

    assert info["version"] == "1.2.0"
    assert info["version"] == __version__
    assert "Muḥāwir" in info["name"]
    assert "ar" in info["supported_languages"]
    assert "en" in info["supported_languages"]
    assert len(info["capabilities"]) >= 8
    assert "stateful_conversation_memory" in info["capabilities"]
    assert "enterprise_guardrails_safety" in info["capabilities"]
    assert "multi_hop_react_reasoning" in info["capabilities"]
    assert info["capabilities"] == SYSTEM_CAPABILITIES
