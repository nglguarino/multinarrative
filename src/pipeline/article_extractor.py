"""
Article-level narrative extraction.

Extracts narratives directly from full articles using multi-agent consensus.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.agents import Agent, MultiAgentConsensus
from ..utils.deduplication import SemanticDedupe


class ArticleNarrativeExtractor:
    """Extract narratives directly from full articles using multi-agent consensus."""

    # CHANGE: New concise prompt for brief narratives
    SYSTEM_PROMPT = """You are an expert analyst extracting political narratives from news articles.

A narrative is a BRIEF, CONCISE claim or framing about actors, events, or issues.

GOOD examples:
- "Discrediting Ukrainian military"
- "Climate movement is alarmist"
- "Praise of Russian President Vladimir Putin"
- "Biden administration is weak on border security"
- "Tech companies threaten privacy"

BAD examples (too verbose):
- "The article discusses how some commentators believe that the Ukrainian military has been ineffective in recent operations"
- "There is a narrative being pushed that suggests climate activists are exaggerating the threat"

Extract 3-7 narratives per article. Each narrative should be:
- Brief (3-10 words)
- Clear and specific
- Focused on a single claim or frame

Output ONLY the narratives, one per line, no numbering or bullets."""

    def __init__(self, agents: List[Agent], embedding_model=None):
        """
        Initialize article extractor.

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
                similarity_threshold=0.85  # Higher threshold for brief narratives
            )
        else:
            self.semantic_dedupe = None

    def extract_from_article(self, article: str, article_id: Any = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract narratives from a full article.

        Args:
            article: Full article text
            article_id: Optional identifier for the article
            metadata: Optional metadata (date, source, actors, etc.)

        Returns:
            Dictionary with article narratives and metadata
        """
        print(f"Processing article {article_id}")

        # Create extraction prompt
        prompt = self._create_extraction_prompt(article, metadata)

        # Get narratives from all agents
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

        # Filter for brevity
        narratives = [n for n in narratives if len(n.split()) >= 2 and len(n.split()) <= 15]

        return {
            'article_id': article_id,
            'narratives': narratives,
            'raw_narrative_count': len(raw_narratives),
            'article_text': article[:500] + "..." if len(article) > 500 else article,
            'metadata': metadata or {}
        }

    def _create_extraction_prompt(self, article: str, metadata: Optional[Dict[str, Any]]) -> str:
        """
        Create prompt for article narrative extraction.

        Args:
            article: Full article text
            metadata: Optional metadata

        Returns:
            Formatted prompt string
        """
        # CHANGE: Include metadata context if available
        context = ""
        if metadata:
            if 'date' in metadata:
                context += f"\nDate: {metadata['date']}"
            if 'source' in metadata:
                context += f"\nSource: {metadata['source']}"
            if 'title' in metadata:
                context += f"\nTitle: {metadata['title']}"

        prompt = f"""Extract the key political narratives from this article.{context}

ARTICLE:
{article}

What are the main narratives or framings present? List 3-7 brief narratives, one per line."""

        return prompt

    def batch_extract(self, articles: List[str], article_ids: List[Any] = None,
                     metadata_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract narratives from multiple articles.

        Args:
            articles: List of article texts
            article_ids: Optional list of article identifiers
            metadata_list: Optional list of metadata dictionaries

        Returns:
            List of article results
        """
        if article_ids is None:
            article_ids = list(range(len(articles)))

        if metadata_list is None:
            metadata_list = [{}] * len(articles)

        results = []
        for article, article_id, metadata in zip(articles, article_ids, metadata_list):
            result = self.extract_from_article(article, article_id, metadata)
            results.append(result)

        return results