# Multi-Agent Narrative Extraction Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for extracting and analyzing political narratives from news articles using multi-agent consensus and graph-based analysis. Designed for research applications requiring large-scale narrative analysis.


## 🌟 Key Features

### Multi-Agent Narrative Extraction
- **5 LLM Agents** working in consensus (GPT-4, Gemini, Claude, Grok, DeepSeek)
- **Hierarchical Processing** from article-level to macro-arguments
- **Semantic Deduplication** using state-of-the-art embeddings
- **GPU Acceleration** for 10-100x speedup

### Graph-Based Analysis
- **Network Construction** linking similar narratives across articles
- **Semantic Clustering** identifying recurring themes
- **Macro-Argument Detection** finding broad narrative patterns
- **Queryable Database** search by actor, topic, date, place

### Interactive Visualization
- **2D/3D Network Graphs** with force-directed layouts
- **t-SNE Clustering** revealing semantic patterns
- **Interactive Dashboard** with real-time filtering
- **Actor Networks** showing co-occurrence relationships
- **Topic Analysis** visualizing distribution patterns

### Production-Ready Pipeline
- **Batch Processing** optimized for thousands of articles
- **Checkpoint System** for fault-tolerant long runs
- **Metadata Extraction** using NLP/ML (spaCy, transformers)
- **Multiple Output Formats** (JSON, CSV, HTML visualizations)

## 📊 Pipeline Overview

```
Input Articles → Metadata Extraction → Multi-Agent Consensus → Graph Construction → Analysis & Visualization
     ↓                ↓                        ↓                      ↓                      ↓
  .txt files    Actors, Topics         3-7 narratives           Network of           Interactive
  + metadata    Places, Dates          per article              narratives            Dashboard
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/narrative-extraction.git
cd narrative-extraction

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python scripts/download_models.py
```

### Configure API Keys

Set your API keys in `config/config.yaml` or as environment variables:

```yaml
api_keys:
  openai: ${OPENAI_API_KEY}
  gemini: ${GEMINI_API_KEY}
  claude: ${CLAUDE_API_KEY}
  grok: ${GROK_API_KEY}
  deepseek: ${DEEPSEEK_API_KEY}
```

### Basic Usage

#### 1. Preprocess Articles

```bash
# Extract metadata from raw text files
python scripts/preprocess_articles.py \
    --input data/input/raw_articles/ \
    --output data/input/articles.json
```

#### 2. Extract Narratives

```bash
# Run the complete extraction pipeline
python scripts/run_extraction.py \
    --input data/input/articles.json \
    --output data/output/
```

#### 3. Visualize Results

```bash
# Generate interactive visualizations
python scripts/visualize_graph.py \
    --graph data/output/narrative_graph.json \
    --output visualizations/
```

#### 4. Query the Graph

```bash
# Search for narratives about specific actors
python scripts/query_graph.py \
    --graph data/output/narrative_graph.json \
    --actor "Trump" \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

## 📝 Example: Google Colab/Jupyter Notebook

```python
import sys
sys.path.insert(0, '/path/to/narrative')

from src.models.agents import create_agent_pool
from src.models.embeddings import EmbeddingMatcher
from src.pipeline.article_extractor import ArticleNarrativeExtractor
from src.pipeline.graph_analyzer import NarrativeGraph
from src.utils.file_loader import ArticleLoader
from src.utils.metadata_extractor import AdvancedMetadataExtractor
from src.visualization.interactive_explorer import create_narrative_dashboard
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load and preprocess articles
articles, ids, metadata = ArticleLoader.load_from_directory('data/input/raw_articles/')

# Extract metadata if needed
metadata_extractor = AdvancedMetadataExtractor(use_gpu=True)
metadata_list = metadata_extractor.batch_extract(articles)

# Initialize pipeline
agents = create_agent_pool(config)
embedding_model = EmbeddingMatcher(device='cuda')
article_extractor = ArticleNarrativeExtractor(agents, embedding_model)

# Extract narratives
article_results = article_extractor.batch_extract(articles, ids, metadata_list)

# Build graph
narrative_graph = NarrativeGraph(embedding_model, similarity_threshold=0.80)
narrative_graph.build_graph(article_results)

# Get insights
across_article = narrative_graph.get_across_article_narratives(min_article_count=3)
summary = narrative_graph.get_summary()

print(f"Extracted {sum(len(r['narratives']) for r in article_results)} narratives")
print(f"Found {len(across_article)} recurring narratives")
print(f"Identified {summary['total_macro_arguments']} macro-arguments")

# Interactive visualization
from src.visualization.graph_visualizer import NarrativeGraphVisualizer
visualizer = NarrativeGraphVisualizer(
    graph_data=narrative_graph.graph_data,
    embedding_model=embedding_model
)
create_narrative_dashboard(visualizer, narrative_graph)
```

## 🎨 Visualization Examples

### Interactive Network Graph
```python
from src.visualization.graph_visualizer import NarrativeGraphVisualizer

visualizer = NarrativeGraphVisualizer(
    graph_path='data/output/narrative_graph.json',
    embedding_model=embedding_model
)

# 2D Network
fig = visualizer.create_2d_visualization(
    color_by='cluster',
    min_article_count=5
)
fig.show()

# t-SNE Clustering
fig = visualizer.create_cluster_view(n_clusters=10)
fig.show()

# Actor Network
fig = visualizer.create_actor_network(min_narratives=10)
fig.show()
```

### Dashboard Interface
```python
from src.visualization.interactive_explorer import create_narrative_dashboard

# Create full dashboard with filters and search
create_narrative_dashboard(visualizer, narrative_graph)
```

## 🔧 Configuration

### Key Parameters

**Deduplication** (`config/config.yaml`)
```yaml
pipeline:
  extraction:
    deduplication_threshold: 0.70  # Within-article similarity (0.65-0.85)
    max_narratives_per_article: 20  # Limit narratives per article
  graph:
    similarity_threshold: 0.80  # Across-article linking (0.75-0.85)
    min_article_count: 3  # Minimum articles for recurring narrative
```

**Metadata Extraction**
```yaml
metadata:
  use_gpu: true  # Enable GPU acceleration
  nlp:
    max_actors: 15
    max_places: 10
    max_topics: 5
```

**Embeddings**
```yaml
embeddings:
  model_name: "all-mpnet-base-v2"  # Sentence transformer model
  device: null  # Auto-detect GPU/CPU
```

## 📊 Performance Benchmarks

| Dataset Size | Processing Time | GPU Acceleration |
|--------------|----------------|------------------|
| 100 articles | ~5 minutes | 10x faster |
| 1,000 articles | ~30 minutes | 20x faster |
| 10,000 articles | ~2 hours | 50x faster |

**Optimizations:**
- Batch processing for metadata extraction (5-10x speedup)
- Async multi-agent calls (4-5x speedup)
- GPU-accelerated NLP models (10-20x speedup)
- Checkpoint system for fault tolerance

## 📈 Research Results

Example results from 400 news articles:

- **5,192 unique narratives** extracted
- **347 recurring narratives** across multiple articles
- **75 macro-arguments** identified
- **Top narrative:** "Climate alarmism is a hoax" (17 articles)
- **Top actor:** Putin (675 narrative mentions)
- **Top topic:** International relations (2,479 narratives)

## 🗂️ Project Structure

```
narrative-extraction-project/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── input/
│   │   ├── raw_articles/        # Input .txt files
│   │   └── articles.json        # Preprocessed articles with metadata
│   └── output/
│       ├── narrative_graph.json # Graph database
│       ├── results.json         # Full results
│       └── *.csv                # CSV exports
├── scripts/
│   ├── download_models.py       # Download NLP models
│   ├── preprocess_articles.py  # Extract metadata
│   ├── run_extraction.py        # Main pipeline
│   ├── query_graph.py           # Query interface
│   └── visualize_graph.py       # Generate visualizations
├── src/
│   ├── models/
│   │   ├── agents.py            # LLM agent implementations
│   │   ├── async_agents.py      # Async multi-agent consensus
│   │   └── embeddings.py        # Embedding-based matching
│   ├── pipeline/
│   │   ├── article_extractor.py # Article-level extraction
│   │   └── graph_analyzer.py    # Graph construction & analysis
│   ├── utils/
│   │   ├── file_loader.py       # File I/O utilities
│   │   ├── metadata_extractor.py # NLP/ML metadata extraction
│   │   ├── deduplication.py     # Semantic deduplication
│   │   └── text_processing.py   # Text utilities
│   └── visualization/
│       ├── graph_visualizer.py  # Plotly visualizations
│       └── interactive_explorer.py # Dashboard interface
├── tests/
│   ├── test_agents.py
│   └── test_utils.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🎯 Use Cases

### Academic Research
- Analyze media framing of political events
- Track narrative evolution over time
- Identify propaganda patterns
- Compare narratives across sources

### Policy Analysis
- Monitor public discourse on key issues
- Identify emerging narratives
- Track actor positioning
- Analyze cross-article consistency

### Media Analysis
- Detect coordinated messaging
- Map narrative networks
- Identify information campaigns
- Track topic evolution

## 🔍 Advanced Features

### Query Interface

```python
# Query by actor
trump_narratives = narrative_graph.query_by_actor("Trump")

# Query by topic
climate_narratives = narrative_graph.query_by_topic("climate")

# Composite query
results = narrative_graph.composite_query(
    actors=["Biden"],
    topics=["economy"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

### Custom Filtering

```python
# Filter by article count
major_narratives = [
    n for n in across_article 
    if n['article_count'] >= 10
]

# Filter by topic
climate_only = narrative_graph.query_by_topic("climate")

# Filter by date range
recent = narrative_graph.query_by_date_range(
    start_date=datetime(2024, 11, 1),
    end_date=datetime(2024, 11, 30)
)
```

### Export Options

```python
# Export graph to JSON
narrative_graph.export_graph('output/graph.json')

# Export to CSV
import csv
with open('narratives.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Narrative', 'Article Count', 'Actors', 'Topics'])
    for node in narrative_graph.nodes.values():
        writer.writerow([
            node.narrative,
            len(node.article_ids),
            ', '.join(node.actors),
            ', '.join(node.topics)
        ])
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Adding New LLM Providers

```python
# In src/models/agents.py
def create_agent_pool(config):
    # Add your custom provider
    if config['api_keys'].get('custom_provider'):
        custom_client = CustomClient(api_key=config['api_keys']['custom_provider'])
        agents.append(Agent(
            name="custom-model",
            client=custom_client,
            provider="custom"
        ))
    return agents
```

### Custom Visualization

```python
from src.visualization.graph_visualizer import NarrativeGraphVisualizer

class CustomVisualizer(NarrativeGraphVisualizer):
    def create_custom_view(self):
        # Your custom visualization logic
        pass
```

## 📚 Documentation

- **[Installation Guide](VISUALIZATION_GUIDE.md)** - Detailed setup instructions
- **[API Reference](docs/API.md)** - Complete API documentation
- **[Examples](examples/)** - Jupyter notebooks with examples
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@software{guarino2025narrative,
  author = {Guarino, Angelo},
  title = {Multi-Agent Narrative Extraction Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/narrative-extraction}
}
```

## 📧 Contact

Angelo Guarino - [GitHub](https://github.com/nglguarino)

Thesis Advisor: Prof. Giovanni Da San Martino - University of Padova

## 🙏 Acknowledgments

- **OpenAI, Anthropic, Google, xAI, DeepSeek** - LLM API providers
- **Hugging Face** - Transformer models and sentence embeddings
- **spaCy** - NLP processing library
- **Plotly** - Interactive visualization library
- **NetworkX** - Graph analysis tools

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This framework is designed for research purposes. Users are responsible for:
- Obtaining appropriate API keys and respecting usage limits
- Ensuring compliance with data privacy regulations
- Validating results for their specific use case
- Proper attribution when using in publications

## 🗺️ Roadmap

- [ ] Support for additional languages (Italian, Spanish, etc.)
- [ ] Temporal analysis of narrative evolution
- [ ] Fine-tuned models for domain-specific extraction
- [ ] Real-time processing pipeline
- [ ] Web interface for non-technical users
- [ ] Integration with social media APIs
- [ ] Advanced sentiment analysis
- [ ] Automated report generation

---

**Built with ❤️ for narrative research**

*Part of thesis work at University of Padova, Italy*
