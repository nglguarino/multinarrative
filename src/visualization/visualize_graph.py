"""
Standalone script to generate narrative graph visualizations.

Usage:
    python scripts/visualize_graph.py --graph data/output/narrative_graph.json --output visualizations/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.graph_visualizer import NarrativeGraphVisualizer
from src.models.embeddings import EmbeddingMatcher


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate interactive visualizations of narrative graphs'
    )
    parser.add_argument('--graph', type=str, required=True,
                       help='Path to narrative_graph.json')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory for HTML files')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Skip embedding-based visualizations (faster but less features)')
    parser.add_argument('--views', type=str, nargs='+',
                       choices=['2d', '3d', 'clusters', 'macros', 'topics', 'actors', 'all'],
                       default=['all'],
                       help='Which visualizations to generate')

    args = parser.parse_args()

    print("=" * 60)
    print("NARRATIVE GRAPH VISUALIZATION")
    print("=" * 60)

    # Load embedding model (unless disabled)
    embedding_model = None
    if not args.no_embeddings:
        print("\nLoading embedding model...")
        try:
            embedding_model = EmbeddingMatcher()
            print("✓ Embedding model loaded")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            print("Proceeding without embedding-based features...")

    # Initialize visualizer
    print(f"\nLoading graph from {args.graph}...")
    visualizer = NarrativeGraphVisualizer(
        graph_path=args.graph,
        embedding_model=embedding_model
    )
    print(f"✓ Loaded graph with {len(visualizer.nodes)} narratives")

    # Create output directory
    import os
    os.makedirs(args.output, exist_ok=True)

    # Determine which views to generate
    views_to_generate = args.views
    if 'all' in views_to_generate:
        views_to_generate = ['2d', '3d', 'clusters', 'macros', 'topics', 'actors']

    print(f"\nGenerating visualizations...")
    print(f"Output directory: {args.output}")

    # Generate visualizations
    try:
        if '2d' in views_to_generate:
            print("\n  Creating 2D network...")
            fig = visualizer.create_2d_visualization(color_by='cluster')
            output_path = os.path.join(args.output, 'network_2d.html')
            fig.write_html(output_path)
            print(f"    ✓ Saved to {output_path}")

        if '3d' in views_to_generate:
            print("\n  Creating 3D network...")
            fig = visualizer.create_3d_visualization(color_by='cluster')
            output_path = os.path.join(args.output, 'network_3d.html')
            fig.write_html(output_path)
            print(f"    ✓ Saved to {output_path}")

        if 'clusters' in views_to_generate:
            if embedding_model:
                print("\n  Creating t-SNE cluster view...")
                fig = visualizer.create_cluster_view()
                output_path = os.path.join(args.output, 'clusters_tsne.html')
                fig.write_html(output_path)
                print(f"    ✓ Saved to {output_path}")
            else:
                print("\n  ⚠️  Skipping cluster view (requires embeddings)")

        if 'macros' in views_to_generate:
            if visualizer.macro_arguments:
                print("\n  Creating macro-arguments view...")
                fig = visualizer.create_macro_argument_view()
                output_path = os.path.join(args.output, 'macro_arguments.html')
                fig.write_html(output_path)
                print(f"    ✓ Saved to {output_path}")
            else:
                print("\n  ⚠️  No macro-arguments found in graph")

        if 'topics' in views_to_generate:
            print("\n  Creating topic distribution...")
            fig = visualizer.create_topic_distribution()
            output_path = os.path.join(args.output, 'topic_distribution.html')
            fig.write_html(output_path)
            print(f"    ✓ Saved to {output_path}")

        if 'actors' in views_to_generate:
            print("\n  Creating actor network...")
            fig = visualizer.create_actor_network()
            output_path = os.path.join(args.output, 'actor_network.html')
            fig.write_html(output_path)
            print(f"    ✓ Saved to {output_path}")

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        return 1

    # Generate index page
    print("\n  Creating index page...")
    create_index_page(args.output, views_to_generate)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\n✓ All visualizations saved to {args.output}/")
    print(f"\nTo view:")
    print(f"  1. Open {args.output}/index.html in a web browser")
    print(f"  2. Or open individual .html files")

    return 0


def create_index_page(output_dir: str, views: list):
    """Create an index.html page with links to all visualizations."""
    import os

    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Narrative Graph Visualizations</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #667eea;
            margin: 0 0 10px 0;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .card {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            text-decoration: none;
            color: inherit;
            transition: all 0.3s ease;
        }
        .card:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        .card h2 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 20px;
        }
        .card p {
            margin: 0;
            color: #666;
            font-size: 14px;
        }
        .icon {
            font-size: 40px;
            margin-bottom: 10px;
        }
        .stats {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Narrative Graph Visualizations</h1>
        <p class="subtitle">Interactive exploration of narrative networks</p>

        <div class="grid">
"""

    # Add cards for each visualization
    viz_info = {
        '2d': ('🕸️', '2D Network', 'Interactive 2D force-directed graph layout', 'network_2d.html'),
        '3d': ('🌐', '3D Network', 'Immersive 3D network visualization', 'network_3d.html'),
        'clusters': ('🎯', 't-SNE Clusters', 'Semantic clustering using t-SNE dimensionality reduction', 'clusters_tsne.html'),
        'macros': ('🎪', 'Macro-Arguments', 'High-level thematic groupings', 'macro_arguments.html'),
        'topics': ('📈', 'Topic Distribution', 'Distribution of topics across narratives', 'topic_distribution.html'),
        'actors': ('👥', 'Actor Network', 'Co-occurrence network of key actors', 'actor_network.html')
    }

    for view_key in views:
        if view_key in viz_info:
            icon, title, description, filename = viz_info[view_key]
            if os.path.exists(os.path.join(output_dir, filename)):
                html_content += f"""
            <a href="{filename}" class="card">
                <div class="icon">{icon}</div>
                <h2>{title}</h2>
                <p>{description}</p>
            </a>
"""

    html_content += """
        </div>

        <div style="margin-top: 40px; padding-top: 30px; border-top: 2px solid #e0e0e0; text-align: center; color: #666;">
            <p>Generated with Narrative Extraction Framework</p>
            <p style="font-size: 12px;">Open any visualization above to explore interactively</p>
        </div>
    </div>
</body>
</html>
"""

    # Write index.html
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"    ✓ Saved to {output_dir}/index.html")


if __name__ == '__main__':
    sys.exit(main())