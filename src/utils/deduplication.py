"""
Deduplication utilities for narrative extraction.

Handles fuzzy matching and semantic deduplication of narratives.
"""

import hashlib
from typing import List, Set, Tuple
from difflib import SequenceMatcher
import numpy as np


class NarrativeDedupe:
    """Deduplicate narratives using fuzzy matching and semantic similarity."""
    
    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Minimum similarity to consider duplicates (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self._seen_hashes = set()
    
    def deduplicate(self, narratives: List[str]) -> List[str]:
        """
        Remove duplicate and near-duplicate narratives.
        
        Args:
            narratives: List of narrative strings
            
        Returns:
            Deduplicated list of narratives
        """
        if not narratives:
            return []
        
        unique_narratives = []
        
        for narrative in narratives:
            narrative = narrative.strip()
            if not narrative:
                continue
            
            # Check if this is a duplicate
            if not self._is_duplicate(narrative, unique_narratives):
                unique_narratives.append(narrative)
        
        return unique_narratives
    
    def _is_duplicate(self, narrative: str, existing: List[str]) -> bool:
        """
        Check if narrative is a duplicate of any existing narratives.
        
        Args:
            narrative: Narrative to check
            existing: List of existing narratives
            
        Returns:
            True if duplicate found
        """
        # Exact match check
        if narrative in existing:
            return True
        
        # Fuzzy match check
        for existing_narrative in existing:
            similarity = self._compute_string_similarity(narrative, existing_narrative)
            if similarity >= self.similarity_threshold:
                return True
        
        return False
    
    def _compute_string_similarity(self, str1: str, str2: str) -> float:
        """
        Compute string similarity using SequenceMatcher.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def hash_narrative(self, narrative: str) -> str:
        """
        Create a hash for a narrative.
        
        Args:
            narrative: Narrative text
            
        Returns:
            MD5 hash string
        """
        return hashlib.md5(narrative.encode('utf-8')).hexdigest()


class SemanticDedupe:
    """Deduplicate narratives using semantic embeddings."""
    
    def __init__(self, embedding_model, similarity_threshold: float = 0.80):
        """
        Initialize semantic deduplicator.
        
        Args:
            embedding_model: EmbeddingMatcher instance
            similarity_threshold: Minimum semantic similarity for duplicates
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(self, narratives: List[str]) -> List[str]:
        """
        Remove semantically duplicate narratives.
        
        Args:
            narratives: List of narrative strings
            
        Returns:
            Deduplicated list of narratives
        """
        if not narratives or len(narratives) <= 1:
            return narratives
        
        # Compute similarity matrix
        similarity_matrix = self.embedding_model.batch_similarity(narratives)
        
        # Find duplicates using similarity threshold
        n = len(narratives)
        keep_indices = set(range(n))
        
        for i in range(n):
            if i not in keep_indices:
                continue
            
            for j in range(i + 1, n):
                if j not in keep_indices:
                    continue
                
                similarity = similarity_matrix[i][j].item()
                if similarity >= self.similarity_threshold:
                    # Keep the shorter one (usually more concise)
                    if len(narratives[i]) <= len(narratives[j]):
                        keep_indices.discard(j)
                    else:
                        keep_indices.discard(i)
                        break
        
        # Return deduplicated narratives
        return [narratives[i] for i in sorted(keep_indices)]


def merge_narrative_lists(list1: List[str], list2: List[str], 
                         threshold: float = 0.75) -> List[str]:
    """
    Merge two lists of narratives, removing duplicates.
    
    Args:
        list1: First list of narratives
        list2: Second list of narratives
        threshold: Similarity threshold for deduplication
        
    Returns:
        Merged and deduplicated list
    """
    dedupe = NarrativeDedupe(similarity_threshold=threshold)
    combined = list1 + list2
    return dedupe.deduplicate(combined)
