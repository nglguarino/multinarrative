# Multi-Agent Narrative Extraction Framework

This repository contains the code for my thesis work (in progress).

## Overview

This framework implements a graph-based narrative extraction system that:
- Extracts concise article-level narratives from news articles using multi-agent consensus
- Creates a graph network of narratives to identify patterns across articles
- Identifies macro-arguments (broad topics of discussion)
- Enables querying by actors, topics, dates, places, and composite queries
- Designed for analyzing thousands of articles for government and policy organizations

## Installation
```bash
# Clone the repository
git clone <repository-url>
cd narrative-extraction-project

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_lg

# Optional: Download all models at once
python scripts/download_models.py
```

### GPU Support (Optional)

For faster processing with GPU:
```bash
# Install PyTorch with CUDA support (check pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Enable GPU in config
# Edit config/config.yaml and set: metadata.use_gpu: true
```

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended for large batches)
- **Disk**: ~3GB for NLP models
- **Python**: 3.8+

## Pipeline Flow
```
Input Articles (with metadata)
     |
     v
1. Article-Level Extraction (Multi-Agent)
     |  - Each agent extracts 3-7 brief narratives
     |  - Consensus aggregation
     |  - Semantic deduplication
     v
2. Graph Construction
     |  - Create nodes for each unique narrative
     |  - Link semantically similar narratives
     |  - Associate metadata (actors, topics, dates, places)
     v
3. Pattern Analysis
     |  - Identify across-article narratives
     |  - Extract macro-arguments via clustering
     |  - Compute graph statistics
     v
4. Queryable Output
     |  - Graph database (JSON)
     |  - CSV exports
     |  - Query API
```

## Usage

### Step 1: Preprocess Raw Articles

Place your .txt files in a directory and extract metadata:
```bash
# Preprocess all .txt files in a directory
python scripts/preprocess_articles.py \
    --input data/input/raw_articles/ \
    --output data/input/articles.json \
    --config config/config.yaml

# Process a single file
python scripts/preprocess_articles.py \
    --input data/input/single_article.txt \
    --output data/input/articles.json
```

### Step 2: Run Narrative Extraction
```bash
python scripts/run_extraction.py \
    --input data/input/articles.json \
    --output data/output/
```

### Step 3: Query the Graph
```bash
# Find all narratives about Trump
python scripts/query_graph.py --graph data/output/narrative_graph.json --actor Trump

# Find narratives about climate during a specific period
python scripts/query_graph.py --graph data/output/narrative_graph.json --topic climate --start-date 2024-01-01 --end-date 2024-12-31

# Composite query
python scripts/query_graph.py \
    --graph data/output/narrative_graph.json \
    --actor Trump \
    --start-date 2024-11-01 \
    --end-date 2024-11-30 \
    --output trump_november_narratives.json
```

## Input Format

Articles should include metadata for full functionality:
```json
[
  {
    "id": 0,
    "text": "Full article text...",
    "metadata": {
      "date": "2024-11-15",
      "source": "New York Times",
      "title": "Article Title",
      "actors": ["Donald Trump", "Joe Biden"],
      "topics": ["election", "economy"],
      "places": ["Pennsylvania", "United States"]
    }
  }
]
```

## API Key Configuration

You need API keys for the LLM providers:
- OpenAI (GPT-4)
- Google Gemini
- Anthropic Claude
- xAI Grok
- DeepSeek

Set them in `config/config.yaml` or as environment variables.

## Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{angeloguarino2025narrative,
  title={Multi-Agent Framework for Narrative Extraction from News Articles},
  author={Angelo Guarino},
  year={2025}
}
```