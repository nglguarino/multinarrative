"""
Paragraph-level narrative extraction.

Extracts narratives from individual paragraphs using multi-agent consensus.
"""

from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..models.agents import Agent, MultiAgentConsensus
from ..models.embeddings import EmbeddingMatcher
from ..utils.text_processing import SmartParagraphSplitter
from ..utils.deduplication import NarrativeDedupe


class ParagraphNarrativeExtractor:
    """Extract narratives from paragraphs using multi-agent consensus."""
    
    # System prompt for narrative extraction
    SYSTEM_PROMPT = """You are an expert analyst specializing in extracting political narratives from news articles.

A narrative is a recurring claim, argument, or framing that shapes how people understand events. Focus on:
- Political positions and claims
- Causal arguments (X causes Y)
- Moral or value judgments
- Attribution of responsibility or blame
- Predictions or warnings

Output ONLY the narratives you identify, one per line. Be concise and specific."""
    
    def __init__(self, agents: List[Agent], embedding_matcher: EmbeddingMatcher = None,
                 batch_size: int = 5, max_workers: int = 3):
        """
        Initialize paragraph extractor.
        
        Args:
            agents: List of LLM agents
            embedding_matcher: Optional embedding matcher for validation
            batch_size: Number of paragraphs to process in parallel
            max_workers: Maximum number of worker threads
        """
        self.agents = agents
        self.consensus = MultiAgentConsensus(agents)
        self.embedding_matcher = embedding_matcher
        self.splitter = SmartParagraphSplitter()
        self.dedupe = NarrativeDedupe(similarity_threshold=0.75)
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def extract_from_article(self, article: str, article_id: Any = None) -> Dict[str, Any]:
        """
        Extract narratives from all paragraphs in an article.
        
        Args:
            article: Full article text
            article_id: Optional identifier for the article
            
        Returns:
            Dictionary with paragraph results and metadata
        """
        # Split into paragraphs
        paragraphs = self.splitter.split_into_paragraphs(article)
        
        print(f"Processing article {article_id}: {len(paragraphs)} paragraphs")
        
        # Process paragraphs in batches
        paragraph_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create futures for all paragraphs
            future_to_para = {}
            for i, paragraph in enumerate(paragraphs):
                future = executor.submit(
                    self._extract_from_paragraph,
                    paragraph, i, article
                )
                future_to_para[future] = (i, paragraph)
            
            # Collect results as they complete
            for future in as_completed(future_to_para):
                para_idx, paragraph = future_to_para[future]
                try:
                    result = future.result()
                    paragraph_results.append(result)
                except Exception as e:
                    print(f"Error processing paragraph {para_idx}: {e}")
                    paragraph_results.append({
                        'paragraph_index': para_idx,
                        'paragraph_text': paragraph,
                        'narratives': [],
                        'confidence': 0.0
                    })
        
        # Sort by paragraph index
        paragraph_results.sort(key=lambda x: x['paragraph_index'])
        
        return {
            'article_id': article_id,
            'paragraph_results': paragraph_results,
            'total_paragraphs': len(paragraphs)
        }
    
    def _extract_from_paragraph(self, paragraph: str, para_index: int, 
                                full_article: str) -> Dict[str, Any]:
        """
        Extract narratives from a single paragraph.
        
        Args:
            paragraph: Paragraph text
            para_index: Index of paragraph in article
            full_article: Full article text for context
            
        Returns:
            Dictionary with paragraph narratives and metadata
        """
        # Create prompt with context
        prompt = self._create_paragraph_prompt(paragraph, full_article)
        
        # Get narratives from all agents
        raw_narratives = self.consensus.generate_with_consensus(
            prompt, 
            self.SYSTEM_PROMPT
        )
        
        # Deduplicate
        narratives = self.dedupe.deduplicate(raw_narratives)
        
        # Compute confidence (simple heuristic: agreement rate)
        confidence = len(narratives) / max(len(raw_narratives), 1)
        
        return {
            'paragraph_index': para_index,
            'paragraph_text': paragraph,
            'narratives': narratives,
            'confidence': confidence,
            'raw_count': len(raw_narratives)
        }
    
    def _create_paragraph_prompt(self, paragraph: str, full_article: str) -> str:
        """
        Create prompt for paragraph narrative extraction.
        
        Args:
            paragraph: Target paragraph
            full_article: Full article for context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Extract political narratives from this paragraph within its article context.

FULL ARTICLE CONTEXT:
{full_article[:2000]}...

TARGET PARAGRAPH:
{paragraph}

What narratives or claims are present in this specific paragraph? List each narrative on a separate line."""
        
        return prompt
    
    def batch_extract(self, articles: List[str], article_ids: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Extract narratives from multiple articles.
        
        Args:
            articles: List of article texts
            article_ids: Optional list of article identifiers
            
        Returns:
            List of article results
        """
        if article_ids is None:
            article_ids = list(range(len(articles)))
        
        results = []
        for article, article_id in zip(articles, article_ids):
            result = self.extract_from_article(article, article_id)
            results.append(result)
        
        return results
