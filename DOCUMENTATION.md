# Narrative Extraction Framework - Technical Documentation

## Architecture Overview

This framework implements a hierarchical bottom-up approach to narrative extraction from news articles using multiple LLM agents for consensus-based analysis.

### Core Components

#### 1. Models (`src/models/`)

**agents.py**
- `Agent`: Base class for LLM interactions, supporting OpenAI, Google Gemini, and Anthropic Claude
- `create_agent_pool()`: Factory function to initialize multiple agents from configuration
- `MultiAgentConsensus`: Manages multi-agent generation and consensus extraction

**embeddings.py**
- `EmbeddingMatcher`: Semantic matching using sentence transformers
- Supports GPU acceleration for efficient batch processing
- Provides similarity computation between texts and narratives

#### 2. Pipeline (`src/pipeline/`)

**paragraph_extractor.py**
- `ParagraphNarrativeExtractor`: Extracts narratives from individual paragraphs
- Uses parallel processing for efficiency
- Includes full article context in prompts for better extraction

**article_aggregator.py**
- `ArticleNarrativeAggregator`: Aggregates paragraph narratives to article level
- Uses semantic deduplication to merge similar narratives
- Identifies overarching themes in articles

**cross_article_analyzer.py**
- `CrossArticleNarrativeAnalyzer`: Identifies narratives spanning multiple articles
- Groups similar narratives using embedding-based clustering
- Tracks narrative frequency and variations across corpus

#### 3. Utilities (`src/utils/`)

**text_processing.py**
- `SmartParagraphSplitter`: Intelligently splits articles into coherent paragraphs
- `normalize_text()`: Text preprocessing and normalization
- `truncate_text()`: Safe text truncation with ellipsis

**deduplication.py**
- `NarrativeDedupe`: Fuzzy string matching for deduplication
- `SemanticDedupe`: Embedding-based semantic deduplication
- `merge_narrative_lists()`: Merge and deduplicate multiple narrative lists

## Pipeline Flow

```
Input Articles
     |
     v
1. Paragraph Splitting (SmartParagraphSplitter)
     |
     v
2. Paragraph-Level Extraction (Multi-Agent)
     |  - Each agent extracts narratives
     |  - Consensus aggregation
     |  - Deduplication
     v
3. Article-Level Aggregation (Multi-Agent)
     |  - Synthesize paragraph narratives
     |  - Identify main themes
     |  - Semantic deduplication
     v
4. Cross-Article Analysis
     |  - Group similar narratives
     |  - Count article frequency
     |  - Identify variations
     v
Output: Hierarchical Narrative Structure
```

## Configuration

### API Keys
Set API keys as environment variables or in `config/config.yaml`:

```bash
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export CLAUDE_API_KEY="your-key"
export GROK_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```

### Model Parameters
Adjust in `config/config.yaml`:
- `temperature`: Controls randomness (0.0-1.0)
- `max_tokens`: Maximum response length
- `similarity_threshold`: Threshold for deduplication (0.0-1.0)

### Pipeline Parameters
- `batch_size`: Number of paragraphs to process in parallel
- `max_workers`: Thread pool size
- `min_article_count`: Minimum articles for overarching narrative

## Usage Examples

### Basic Usage

```python
from src.models.agents import create_agent_pool
from src.models.embeddings import EmbeddingMatcher
from src.pipeline.paragraph_extractor import ParagraphNarrativeExtractor
from src.pipeline.article_aggregator import ArticleNarrativeAggregator
from src.pipeline.cross_article_analyzer import CrossArticleNarrativeAnalyzer

# Initialize
agents = create_agent_pool()
embeddings = EmbeddingMatcher()

paragraph_extractor = ParagraphNarrativeExtractor(agents, embeddings)
article_aggregator = ArticleNarrativeAggregator(agents, embeddings)
cross_article_analyzer = CrossArticleNarrativeAnalyzer(embeddings)

# Process
articles = ["article 1 text...", "article 2 text..."]
paragraph_results = paragraph_extractor.batch_extract(articles)
article_results = article_aggregator.batch_aggregate(paragraph_results)
cross_article_results = cross_article_analyzer.analyze(article_results)
```

### Command Line

```bash
# Run full pipeline
python scripts/run_extraction.py \
    --input data/input/articles.json \
    --output data/output/ \
    --config config/config.yaml

# Analyze results
python scripts/evaluate_results.py \
    --results data/output/results.json \
    --export-freq data/output/narrative_frequency.csv
```

## Input Format

Articles should be in JSON format:

```json
[
  {
    "id": 0,
    "text": "Full article text..."
  },
  {
    "id": 1,
    "text": "Another article..."
  }
]
```

Or as a simple list:

```json
[
  "Article 1 text...",
  "Article 2 text..."
]
```

## Output Format

### Main Results (results.json)
```json
{
  "article_results": [
    {
      "article_id": 0,
      "article_narratives": ["narrative 1", "narrative 2"],
      "paragraph_results": [
        {
          "paragraph_index": 0,
          "paragraph_text": "...",
          "narratives": ["para narrative 1"],
          "confidence": 0.85
        }
      ]
    }
  ],
  "cross_article": {
    "overarching_narratives": [
      {
        "narrative": "Main overarching narrative",
        "article_count": 5,
        "article_ids": [0, 1, 2, 3, 4],
        "variations": ["variation 1", "variation 2"]
      }
    ]
  },
  "summary": {
    "total_articles": 10,
    "total_paragraphs": 100,
    "total_paragraph_narratives": 250,
    "total_article_narratives": 85,
    "overarching_narratives": 20
  }
}
```

### CSV Output (results.csv)
Flattened paragraph-level data for analysis in spreadsheet tools.

### Hierarchy JSON (results_hierarchy.json)
Structured format showing paragraph → article → overarching narrative relationships.

## Performance Optimization

### GPU Usage
The embedding model automatically uses GPU if available:
```python
embeddings = EmbeddingMatcher(device='cuda')  # Force GPU
embeddings = EmbeddingMatcher(device='cpu')   # Force CPU
```

### Parallel Processing
Adjust workers and batch size:
```python
paragraph_extractor = ParagraphNarrativeExtractor(
    agents, 
    batch_size=10,      # Process 10 paragraphs at a time
    max_workers=5       # Use 5 worker threads
)
```

### Rate Limiting
Configure in `config/config.yaml`:
```yaml
processing:
  rate_limit:
    requests_per_minute: 50
    retry_attempts: 3
    retry_delay: 2
```

## Testing

Run tests with pytest:
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=src tests/
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch size in config
- Use smaller embedding model
- Switch to CPU processing

**2. API Rate Limits**
- Reduce `batch_size` and `max_workers`
- Add delays between requests
- Use fewer agents

**3. Poor Narrative Quality**
- Adjust prompt templates in extractors
- Increase similarity thresholds
- Use more diverse agent pool

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourname2025narrative,
  title={Bottom-Up Multi-Agent Framework for Narrative Extraction from News Articles},
  author={Your Name},
  booktitle={Proceedings of the Association for Computational Linguistics},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details
