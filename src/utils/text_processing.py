"""
Text processing utilities for narrative extraction.

Includes paragraph splitting and text normalization.
"""

from typing import List
import re


class SmartParagraphSplitter:
    """Split articles into coherent paragraphs with smart merging."""
    
    def __init__(self, min_length: int = 100):
        """
        Initialize paragraph splitter.
        
        Args:
            min_length: Minimum paragraph length in characters
        """
        self.min_length = min_length
    
    def split_into_paragraphs(self, article: str) -> List[str]:
        """
        Split article into paragraphs with smart merging.
        
        Args:
            article: Full article text
            
        Returns:
            List of paragraph strings
        """
        # Initial split on double newlines
        raw_paragraphs = article.split('\n\n')
        
        # Merge short paragraphs
        merged_paragraphs = []
        buffer = ""
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if buffer:
                buffer += " " + para
            else:
                buffer = para
            
            if len(buffer) >= self.min_length:
                merged_paragraphs.append(buffer)
                buffer = ""
        
        # Add remaining buffer if it meets minimum length
        if buffer and len(buffer) >= self.min_length:
            merged_paragraphs.append(buffer)
        
        return merged_paragraphs


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and standardizing format.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length, adding ellipsis if needed.
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
