"""
Article-level narrative aggregation.

Aggregates paragraph-level narratives into article-level summaries.
"""

from typing import List, Dict, Any
from ..models.agents import Agent, MultiAgentConsensus
from ..utils.deduplication import SemanticDedupe


class ArticleNarrativeAggregator:
    """Aggregate paragraph narratives into article-level narratives."""
    
    SYSTEM_PROMPT = """You are an expert at identifying overarching political narratives across multiple text segments.

Your task is to synthesize paragraph-level narratives into broader article-level narratives. Look for:
- Common themes across paragraphs
- Main arguments or claims of the article
- Recurring patterns or framings

Output ONLY the synthesized narratives, one per line. Be concise and capture the essence."""
    
    def __init__(self, agents: List[Agent], embedding_model=None):
        """
        Initialize article aggregator.
        
        Args:
            agents: List of LLM agents
            embedding_model: Optional embedding model for semantic deduplication
        """
        self.agents = agents
        self.consensus = MultiAgentConsensus(agents)
        self.embedding_model = embedding_model
        if embedding_model:
            self.semantic_dedupe = SemanticDedupe(
                embedding_model, 
                similarity_threshold=0.80
            )
        else:
            self.semantic_dedupe = None
    
    def aggregate_article(self, paragraph_results: Dict[str, Any]) -> List[str]:
        """
        Aggregate paragraph narratives into article-level narratives.
        
        Args:
            paragraph_results: Results from ParagraphNarrativeExtractor
            
        Returns:
            List of article-level narrative strings
        """
        # Collect all paragraph narratives
        all_paragraph_narratives = []
        for para_result in paragraph_results['paragraph_results']:
            all_paragraph_narratives.extend(para_result['narratives'])
        
        if not all_paragraph_narratives:
            return []
        
        # Create aggregation prompt
        prompt = self._create_aggregation_prompt(all_paragraph_narratives)
        
        # Get article-level narratives from agents
        raw_narratives = self.consensus.generate_with_consensus(
            prompt,
            self.SYSTEM_PROMPT
        )
        
        # Deduplicate semantically if embedding model available
        if self.semantic_dedupe:
            narratives = self.semantic_dedupe.deduplicate(raw_narratives)
        else:
            # Fall back to basic deduplication
            narratives = list(set(raw_narratives))
        
        return narratives
    
    def _create_aggregation_prompt(self, paragraph_narratives: List[str]) -> str:
        """
        Create prompt for article-level aggregation.
        
        Args:
            paragraph_narratives: List of paragraph-level narratives
            
        Returns:
            Formatted prompt string
        """
        narratives_text = "\n".join([f"- {n}" for n in paragraph_narratives])
        
        prompt = f"""Given these narratives extracted from different paragraphs of an article, identify the overarching article-level narratives.

PARAGRAPH-LEVEL NARRATIVES:
{narratives_text}

Synthesize these into broader, article-level narratives. Each narrative should capture a main theme or argument of the article. List one narrative per line."""
        
        return prompt
    
    def batch_aggregate(self, paragraph_results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate narratives for multiple articles.
        
        Args:
            paragraph_results_list: List of paragraph extraction results
            
        Returns:
            List of dictionaries with article narratives
        """
        results = []
        
        for para_results in paragraph_results_list:
            article_narratives = self.aggregate_article(para_results)
            
            results.append({
                'article_id': para_results['article_id'],
                'article_narratives': article_narratives,
                'paragraph_results': para_results['paragraph_results']
            })
        
        return results
