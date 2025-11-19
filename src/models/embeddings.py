"""
Embedding-based semantic matching for narrative extraction.

Uses sentence transformers for dense vector embeddings and similarity computation.
"""

import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple


class EmbeddingMatcher:
    """
    Matches paragraphs to narratives using dense vector embeddings.
    Optimized for GPU usage with high-quality models.
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', device: str = None):
        """
        Initialize embedding matcher.
        
        Args:
            model_name: Name of the sentence transformer model
            device: 'cuda' or 'cpu', auto-detected if None
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading embedding model: {model_name} on {self.device.upper()}...")
        
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)
        print("Model loaded successfully.")
    
    def match_paragraph_to_narratives(self, paragraph: str, narratives: List[str], 
                                     threshold: float = 0.40) -> List[Tuple[str, float]]:
        """
        Semantic search to find narratives that match the paragraph's meaning.
        
        Args:
            paragraph: The text segment to analyze
            narratives: A list of narrative statements to check against
            threshold: Minimum similarity score (0-1)
            
        Returns:
            A sorted list of tuples: (narrative_text, similarity_score)
        """
        if not paragraph or not narratives:
            return []
        
        # Encode the paragraph (Query)
        para_embedding = self.model.encode(paragraph, convert_to_tensor=True)
        
        # Encode all narratives (Corpus)
        narrative_embeddings = self.model.encode(narratives, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarities = util.cos_sim(para_embedding, narrative_embeddings)[0]
        
        # Filter by threshold and sort
        matches = []
        for i, score in enumerate(similarities):
            score_val = score.item()
            if score_val >= threshold:
                matches.append((narratives[i], score_val))
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        
        similarity = util.cos_sim(emb1, emb2)[0][0]
        return similarity.item()
    
    def batch_similarity(self, texts: List[str]) -> torch.Tensor:
        """
        Compute pairwise similarities for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Similarity matrix (len(texts) x len(texts))
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        similarities = util.cos_sim(embeddings, embeddings)
        return similarities
