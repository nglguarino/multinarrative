"""
Article-level narrative extraction.

Extracts narratives directly from full articles using multi-agent consensus.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.agents import Agent, MultiAgentConsensus
from ..models.async_agents import AsyncMultiAgentConsensus
from ..utils.deduplication import SemanticDedupe


class ArticleNarrativeExtractor:
    """Extract narratives directly from full articles using multi-agent consensus."""

    # CHANGE: Completely overhauled to target "Gold Label" abstraction level.
    # The goal is no longer 'Extraction' (finding specific text), but 'Abstraction' (classifying themes).
    SYSTEM_PROMPT = """You are an expert Narrative Analyst. Your task is to identify the high-level NARRATIVE THEMES present in the text.

        CRITICAL DISTINCTION:
        Do NOT extract specific headlines or event details.
        Do NOT output specific arguments like "The Green Deal is failing."
        INSTEAD, output the GENERALIZED THEME: "Climate policies are ineffective."

        Your goal is to bridge the gap between specific instances and abstract categories.

        INSTRUCTIONS:
        1. Identify specific argumentative claims in the text.
        2. GENERALIZE them into a standardized narrative label.
        3. Remove specific entity names (e.g., replace "Biden" with "The West/Elites", replace "Volkswagen" with "Industry").
        4. Use the present tense (General Truth).

        TARGET CATEGORIES & FORMAT (Map specific claims to these types of General Themes):

        [CATEGORY: INCOMPETENCE & FAILURE]
        - Specific: "German industry is collapsing due to gas prices."
        -> GOLD LABEL: "Sanctions imposed by Western countries will backfire"
        - Specific: "Wind turbines are freezing in Texas."
        -> GOLD LABEL: "Renewable energy is unreliable"

        [CATEGORY: CORRUPTION & MALICE]
        - Specific: "Zelensky is buying yachts with aid money."
        -> GOLD LABEL: "Ukrainian government is corrupt"
        - Specific: "The WEF is orchestrating the energy crisis."
        -> GOLD LABEL: "Global elites are conspiring against the public"

        [CATEGORY: THREATS & FEAR]
        - Specific: "Sending F-16s will lead to nuclear war."
        -> GOLD LABEL: "Western intervention risks World War III"
        - Specific: "We will eat bugs and own nothing."
        -> GOLD LABEL: "Green agenda threatens individual freedom"

        [CATEGORY: DENIAL & DOWNPLAYING]
        - Specific: "The ice in Antarctica is actually growing."
        -> GOLD LABEL: "Climate change indicators are not alarming"
        - Specific: "Russia only attacked to stop NATO expansion."
        -> GOLD LABEL: "Russia is acting in self-defense"

        OUTPUT REQUIREMENTS:
        - Output ONLY the generalized narrative themes.
        - One theme per line.
        - Keep sentences short, standardized, and abstract.
        """

    def __init__(self, agents: List[Agent], embedding_model=None, max_narratives: int = None):
        """
        Initialize article extractor.

        Args:
            agents: List of LLM agents
            embedding_model: Optional embedding model for semantic deduplication
        """
        self.agents = agents
        self.consensus = AsyncMultiAgentConsensus(agents)
        self.embedding_model = embedding_model
        self.max_narratives = max_narratives
        if embedding_model:
            # CHANGE: Increased threshold slightly as abstract themes are more likely to overlap
            self.semantic_dedupe = SemanticDedupe(
                embedding_model,
                similarity_threshold=0.90
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
            narratives = list(set(raw_narratives))

        # Apply max narratives limit if configured
        if self.max_narratives:
            narratives = narratives[:self.max_narratives]

        # CHANGE: Removed the "brevity filter" that was deleting short/long narratives.
        # Abstract labels can be short ("The West is weak") or long ("Sanctions imposed by Western countries will backfire").
        # We rely on the prompt to control length.

        return {
            'article_id': article_id,
            'narratives': narratives,
            'raw_narrative_count': len(raw_narratives),
            'article_text': article[:500] + "..." if len(article) > 500 else article,
            'metadata': metadata or {}
        }

    def _create_extraction_prompt(self, article: str, metadata: Optional[Dict[str, Any]]) -> str:
        # CHANGE: Added explicit instruction in the user prompt to reinforce the system prompt
        context = ""
        if metadata:
            if 'date' in metadata:
                context += f"\nDate: {metadata['date']}"
            if 'source' in metadata:
                context += f"\nSource: {metadata['source']}"
            if 'title' in metadata:
                context += f"\nTitle: {metadata['title']}"

        prompt = f"""Analyze the following article and extract the underlying NARRATIVE THEMES.
        
        Remember: Do not describe *what* happened (events). Describe *the argument* being made (themes).
        Map specific instances to general categories (e.g., "Germany's economy" -> "The West").

    ARTICLE:
    {article}
    """

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