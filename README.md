# Multi-Agent Narrative Extraction Framework

This repository contains the implementation for the paper submitted to ACL: "Bottom-Up Multi-Agent Framework for Narrative Extraction from News Articles"

## Overview

This framework implements a hierarchical narrative extraction system that:
- Extracts paragraph-level narratives from news articles
- Aggregates narratives at the article level
- Identifies overarching narratives across multiple articles
- Uses multiple LLM agents for consensus-based extraction

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd narrative-extraction-project

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
narrative-extraction-project/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml           # Configuration file for API keys and parameters
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agents.py         # LLM agent implementations
│   │   └── embeddings.py     # Embedding-based matching
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── paragraph_extractor.py
│   │   ├── article_aggregator.py
│   │   └── cross_article_analyzer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_processing.py
│   │   └── deduplication.py
│   └── visualization/
│       ├── __init__.py
│       └── analysis.py
├── scripts/
│   ├── run_extraction.py     # Main execution script
│   └── evaluate_results.py   # Evaluation and analysis
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_pipeline.py
│   └── test_utils.py
└── data/
    ├── input/                # Place input articles here
    └── output/               # Results will be saved here
```

## Usage

### Basic Usage

```python
from src.pipeline.paragraph_extractor import ParagraphNarrativeExtractor
from src.pipeline.article_aggregator import ArticleNarrativeAggregator
from src.pipeline.cross_article_analyzer import CrossArticleNarrativeAnalyzer
from src.models.agents import create_agent_pool

# Initialize components
agents = create_agent_pool()
paragraph_extractor = ParagraphNarrativeExtractor(agents)
article_aggregator = ArticleNarrativeAggregator(agents)
cross_article_analyzer = CrossArticleNarrativeAnalyzer(agents)

# Process articles
articles = ["article text 1", "article text 2", ...]
results = process_articles(articles, paragraph_extractor, article_aggregator, cross_article_analyzer)
```

### Running the Full Pipeline

```bash
python scripts/run_extraction.py --input data/input/articles.json --output data/output/
```

### Configuration

Edit `config/config.yaml` to set:
- API keys for different LLM providers
- Model parameters (temperature, max_tokens, etc.)
- Pipeline parameters (thresholds, batch sizes, etc.)

## API Key Configuration

You need API keys for the following services:
- OpenAI (GPT-4)
- Google Gemini
- Anthropic Claude
- xAI Grok
- DeepSeek

Set them in `config/config.yaml` or as environment variables.

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

## License

MIT License
