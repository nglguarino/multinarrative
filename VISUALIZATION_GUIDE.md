# Interactive Graph Visualization System

## Overview

This visualization system provides interactive exploration of your narrative graphs with multiple views, filtering, and clustering capabilities.

## Installation

### 1. Update your requirements.txt

Replace your current `requirements.txt` with the updated version that includes visualization dependencies:

```bash
# In your repository root
cp requirements_updated.txt requirements.txt
pip install -r requirements.txt
```

### 2. Add visualization modules to your repository

Copy these files to your repository structure:

```
narrative-extraction-project/
├── src/
│   └── visualization/
│       ├── __init__.py (create empty file)
│       ├── graph_visualizer.py
│       └── interactive_explorer.py
└── scripts/
    └── visualize_graph.py
```

## Usage

### Method 1: Interactive Notebook (Recommended)

Add this cell to your Colab/Jupyter notebook after building the graph:

```python
# Import visualization modules
from src.visualization.graph_visualizer import NarrativeGraphVisualizer
from src.visualization.interactive_explorer import create_narrative_dashboard
from src.models.embeddings import EmbeddingMatcher

# Load embedding model
embedding_model = EmbeddingMatcher(
    model_name='all-mpnet-base-v2',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Initialize visualizer
visualizer = NarrativeGraphVisualizer(
    graph_path='data/output/narrative_graph.json',
    embedding_model=embedding_model
)

# Create interactive dashboard
create_narrative_dashboard(visualizer, narrative_graph)
```

### Method 2: Generate Static HTML Files

From command line:

```bash
# Generate all visualizations
python scripts/visualize_graph.py \
    --graph data/output/narrative_graph.json \
    --output visualizations/

# Generate specific views only
python scripts/visualize_graph.py \
    --graph data/output/narrative_graph.json \
    --output visualizations/ \
    --views 2d 3d clusters

# Skip embeddings (faster but limited features)
python scripts/visualize_graph.py \
    --graph data/output/narrative_graph.json \
    --output visualizations/ \
    --no-embeddings
```

Then open `visualizations/index.html` in a web browser.

### Method 3: Programmatic Use

```python
from src.visualization.graph_visualizer import NarrativeGraphVisualizer

# Initialize
visualizer = NarrativeGraphVisualizer(
    graph_path='data/output/narrative_graph.json'
)

# Create 2D network
fig = visualizer.create_2d_visualization(
    color_by='cluster',
    min_article_count=3
)
fig.show()  # In notebook
# OR
fig.write_html('my_network.html')  # Save to file
```

## Features

### 1. Interactive Network Visualizations

**2D Network**
- Force-directed graph layout
- Interactive zoom, pan
- Node size = article count
- Color by cluster/topic/count
- Hover for details

**3D Network**
- Full 3D rotation
- Immersive exploration
- Same coloring options

### 2. Clustering Analysis

**t-SNE Visualization**
- Semantic clustering using KMeans
- 2D projection of high-dimensional embeddings
- Color-coded clusters
- Reveals thematic groupings

### 3. Statistical Views

**Macro-Arguments**
- Bar chart of broad themes
- Narrative count vs article count
- Top 20 topics

**Topic Distribution**
- Horizontal bar chart
- Top N topics by frequency
- Color-coded by count

**Actor Network**
- Co-occurrence graph
- Shows which actors appear together
- Edge thickness = co-occurrence count

### 4. Interactive Dashboard

The dashboard provides:
- **Filters**: Search, topic, actor, article count
- **Multiple Views**: Switch between visualizations
- **Query Interface**: Search by actor/topic/place
- **Export**: Download filtered results as JSON
- **Real-time Updates**: Dynamic filtering

## Visualization Options

### Coloring Schemes

- `'cluster'`: Semantic clusters (requires embeddings)
- `'topic'`: Primary topic category
- `'article_count'`: Number of articles (heatmap)

### Layout Algorithms

For 2D networks:
- `'spring'`: Force-directed (default, best for most cases)
- `'kamada_kawai'`: Energy minimization (good for small graphs)
- `'circular'`: Circular arrangement (good for hierarchies)

### Filtering

All visualizations support:
- **Min article count**: Show only narratives in N+ articles
- **Topic filter**: Show only specific topics
- **Actor filter**: Show only specific actors
- **Search**: Text search in narratives

## Examples

### Example 1: Focus on Climate Narratives

```python
# In notebook dashboard:
# 1. Go to "Interactive Explorer" tab
# 2. Select "climate/environment" in Topics filter
# 3. Set Min Articles to 5
# 4. Click "Update Visualization"
```

### Example 2: Explore Trump Narratives

```python
# In notebook dashboard:
# 1. Go to "Query Interface" tab
# 2. Select "By Actor"
# 3. Enter "Trump"
# 4. Click "Run Query"
```

### Example 3: Generate Publication-Ready Figures

```python
# Create high-quality 2D network
fig = visualizer.create_2d_visualization(
    color_by='cluster',
    min_article_count=5,
    n_clusters=8
)

# Customize for publication
fig.update_layout(
    font=dict(size=14, family='Arial'),
    width=1200,
    height=800
)

# Save as static image (requires kaleido)
fig.write_image('figures/narrative_network.png', scale=2)

# Or save as HTML
fig.write_html('figures/narrative_network.html')
```

## Performance Tips

1. **For large graphs (>1000 nodes)**:
   - Use `min_article_count` filter
   - Start with 2D before 3D
   - Consider `--no-embeddings` for faster loading

2. **For best clustering**:
   - Always use embeddings
   - Adjust `n_clusters` parameter
   - Try different perplexity values for t-SNE

3. **For presentations**:
   - Use 3D networks for "wow factor"
   - Use 2D networks for clarity
   - Export filtered views as separate HTML files

## Troubleshooting

**Issue**: "No module named 'plotly'"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Visualizations not loading in notebook
- **Solution**: Restart kernel and re-run imports

**Issue**: "Embeddings required for cluster view"
- **Solution**: Make sure to pass `embedding_model` when creating visualizer

**Issue**: Graph too crowded
- **Solution**: Increase `min_article_count` filter or use t-SNE view

**Issue**: Slow performance
- **Solution**: Use `--no-embeddings` or filter to smaller subset

## Next Steps

1. **Customize visualizations**: Modify colors, sizes, layouts in `graph_visualizer.py`
2. **Add new views**: Create custom visualization functions
3. **Integrate with analysis**: Combine with statistical analysis
4. **Export for publication**: Generate high-quality static images

## File Structure

After installation, your project should look like:

```
narrative-extraction-project/
├── src/
│   └── visualization/
│       ├── __init__.py
│       ├── graph_visualizer.py      # Core visualization logic
│       └── interactive_explorer.py  # Dashboard and widgets
├── scripts/
│   └── visualize_graph.py           # CLI tool
├── visualizations/                  # Generated HTML files
│   ├── index.html
│   ├── network_2d.html
│   ├── network_3d.html
│   ├── clusters_tsne.html
│   ├── macro_arguments.html
│   ├── topic_distribution.html
│   └── actor_network.html
└── requirements.txt                 # Updated dependencies
```

## Citation

If you use these visualizations in your research, please cite appropriately.