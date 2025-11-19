"""
Tests for text processing utilities.
"""

import pytest
from src.utils.text_processing import (
    SmartParagraphSplitter, 
    normalize_text, 
    truncate_text
)


def test_paragraph_splitting():
    """Test paragraph splitting functionality."""
    splitter = SmartParagraphSplitter(min_length=50)
    
    article = """This is the first paragraph. It contains some text.

This is the second paragraph with more content.

Short.

This is the third paragraph which is longer."""
    
    paragraphs = splitter.split_into_paragraphs(article)
    
    # Should merge short paragraphs
    assert len(paragraphs) >= 2
    # Each paragraph should meet minimum length
    for para in paragraphs:
        assert len(para) >= 50


def test_normalize_text():
    """Test text normalization."""
    text = "This   has   extra    spaces\n\nand newlines"
    normalized = normalize_text(text)
    
    assert "  " not in normalized
    assert "\n" not in normalized
    assert normalized == "This has extra spaces and newlines"


def test_truncate_text():
    """Test text truncation."""
    text = "A" * 300
    truncated = truncate_text(text, max_length=100)
    
    assert len(truncated) <= 100
    assert truncated.endswith("...")
    
    # Short text should not be truncated
    short_text = "Short"
    assert truncate_text(short_text, max_length=100) == short_text


if __name__ == '__main__':
    pytest.main([__file__])
