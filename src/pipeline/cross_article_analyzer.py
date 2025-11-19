"""
Cross-article narrative analysis.

Identifies overarching narratives that appear across multiple articles.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from ..models.embeddings import EmbeddingMatcher
from ..utils.deduplication import SemanticDedupe


class CrossArticleNarrativeAnalyzer:
    """Identify narratives that span multiple articles."""
    
    def __init__(self, embedding_model: EmbeddingMatcher, 
                 min_article_count: int = 2,
                 similarity_threshold: float = 0.80):
        """
        Initialize cross-article analyzer.
        
        Args:
            embedding_model: Embedding model for semantic matching
            min_article_count: Minimum articles for a narrative to be "overarching"
            similarity_threshold: Similarity threshold for grouping narratives
        """
        self.embedding_model = embedding_model
        self.min_article_count = min_article_count
        self.similarity_threshold = similarity_threshold
        self.semantic_dedupe = SemanticDedupe(embedding_model, similarity_threshold)
    
    def analyze(self, article_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze narratives across all articles.
        
        Args:
            article_results: List of article results from ArticleNarrativeAggregator
            
        Returns:
            Dictionary with overarching narratives and cross-article statistics
        """
        # Collect all article-level narratives with their sources
        narrative_to_articles = defaultdict(list)
        
        for article_result in article_results:
            article_id = article_result['article_id']
            for narrative in article_result['article_narratives']:
                narrative_to_articles[narrative].append(article_id)
        
        # Group similar narratives
        overarching_narratives = self._group_similar_narratives(
            narrative_to_articles
        )
        
        # Filter by minimum article count
        overarching_narratives = [
            n for n in overarching_narratives 
            if n['article_count'] >= self.min_article_count
        ]
        
        # Sort by article count (descending)
        overarching_narratives.sort(key=lambda x: x['article_count'], reverse=True)
        
        return {
            'overarching_narratives': overarching_narratives,
            'total_articles': len(article_results),
            'total_unique_narratives': len(narrative_to_articles)
        }
    
    def _group_similar_narratives(self, narrative_to_articles: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Group semantically similar narratives together.
        
        Args:
            narrative_to_articles: Mapping of narratives to article IDs
            
        Returns:
            List of grouped narrative dictionaries
        """
        narratives = list(narrative_to_articles.keys())
        if not narratives:
            return []
        
        # Compute similarity matrix
        similarity_matrix = self.embedding_model.batch_similarity(narratives)
        
        # Group similar narratives
        n = len(narratives)
        groups = []  # List of groups, each group is a list of narrative indices
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Start new group with this narrative
            group = [i]
            assigned.add(i)
            
            # Find similar narratives
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                similarity = similarity_matrix[i][j].item()
                if similarity >= self.similarity_threshold:
                    group.append(j)
                    assigned.add(j)
            
            groups.append(group)
        
        # Create output format
        result = []
        for group in groups:
            # Representative narrative (first one in group)
            main_narrative = narratives[group[0]]
            
            # Collect all article IDs
            all_article_ids = set()
            variations = []
            
            for idx in group:
                narrative = narratives[idx]
                all_article_ids.update(narrative_to_articles[narrative])
                if idx > 0:  # Add variations (skip main narrative)
                    variations.append(narrative)
            
            result.append({
                'narrative': main_narrative,
                'article_count': len(all_article_ids),
                'article_ids': sorted(list(all_article_ids)),
                'variations': variations[:5]  # Limit to 5 variations
            })
        
        return result
    
    def get_narrative_coverage(self, article_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate coverage statistics for narratives.
        
        Args:
            article_results: List of article results
            
        Returns:
            Dictionary with coverage statistics
        """
        total_articles = len(article_results)
        articles_with_narratives = sum(
            1 for r in article_results 
            if r['article_narratives']
        )
        
        total_narratives = sum(
            len(r['article_narratives']) 
            for r in article_results
        )
        
        return {
            'total_articles': total_articles,
            'articles_with_narratives': articles_with_narratives,
            'coverage_rate': articles_with_narratives / max(total_articles, 1),
            'total_narratives': total_narratives,
            'avg_narratives_per_article': total_narratives / max(total_articles, 1)
        }
