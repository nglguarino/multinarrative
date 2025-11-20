"""
Main execution script for narrative extraction pipeline.

Usage:
    python scripts/run_extraction.py --input data/input/articles.json --output data/output/
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.agents import create_agent_pool
from src.models.embeddings import EmbeddingMatcher
from src.pipeline.article_extractor import ArticleNarrativeExtractor
from src.pipeline.graph_analyzer import NarrativeGraph


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expand environment variables in API keys
    for key in config.get('api_keys', {}):
        value = config['api_keys'][key]
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            config['api_keys'][key] = os.getenv(env_var)

    return config


def load_articles(input_path: str) -> tuple:
    """
    Load articles from input file or directory.

    Supports:
    - JSON file with structured data
    - Directory with .txt files (will extract metadata on-the-fly)
    - Single .txt file
    """
    from src.utils.file_loader import ArticleLoader

    input_path_obj = Path(input_path)

    # If it's a JSON file, load normally
    if input_path_obj.is_file() and input_path_obj.suffix == '.json':
        articles, ids, metadata, _ = ArticleLoader.load_from_json(input_path)
        return articles, ids, metadata

    # If it's a directory or .txt file, we need to extract metadata
    elif input_path_obj.is_dir() or (input_path_obj.is_file() and input_path_obj.suffix == '.txt'):
        print(f"Warning: Loading raw .txt file(s) without pre-extracted metadata.")
        print(f"For better performance, consider running preprocess_articles.py first.")
        print(f"Continuing with on-the-fly metadata extraction...\n")

        if input_path_obj.is_file():
            text, filename = ArticleLoader.load_from_txt_file(input_path)
            articles = [text]
            filenames = [filename]
            ids = [0]
        else:
            articles, filenames, ids = ArticleLoader.load_from_directory(input_path)

        # Extract metadata on-the-fly
        from src.utils.metadata_extractor import AdvancedMetadataExtractor

        # Load config for GPU settings
        config = load_config()
        use_gpu = config.get('metadata', {}).get('use_gpu', False)

        extractor = AdvancedMetadataExtractor(use_gpu=use_gpu)
        metadata = extractor.batch_extract(articles, filenames)

        return articles, ids, metadata

    else:
        raise ValueError(f"Invalid input: {input_path}. Expected JSON file, .txt file, or directory.")


def save_results(results: dict, graph: NarrativeGraph, output_dir: str):
    """Save results to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save full results as JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved results to {json_path}")

    # Save graph
    graph_path = os.path.join(output_dir, 'narrative_graph.json')
    graph.export_graph(graph_path)

    # Save CSV exports
    export_narratives_csv(results, os.path.join(output_dir, 'narratives.csv'))
    export_across_articles_csv(results['across_article_narratives'],
                               os.path.join(output_dir, 'across_article_narratives.csv'))
    export_macro_arguments_csv(results['macro_arguments'],
                               os.path.join(output_dir, 'macro_arguments.csv'))


def export_narratives_csv(results: dict, output_path: str):
    """Export all narratives to CSV."""
    import csv

    rows = []
    for article_result in results['article_results']:
        article_id = article_result['article_id']
        metadata = article_result.get('metadata', {})

        for narrative in article_result['narratives']:
            rows.append({
                'article_id': article_id,
                'narrative': narrative,
                'date': metadata.get('date', ''),
                'source': metadata.get('source', ''),
                'actors': ', '.join(metadata.get('actors', [])),
                'topics': ', '.join(metadata.get('topics', [])),
                'places': ', '.join(metadata.get('places', []))
            })

    if rows:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Exported {len(rows)} narratives to {output_path}")


def export_across_articles_csv(across_article_narratives: list, output_path: str):
    """Export across-article narratives to CSV."""
    import csv

    if not across_article_narratives:
        return

    rows = []
    for item in across_article_narratives:
        rows.append({
            'narrative': item['primary_narrative'],
            'article_count': item['article_count'],
            'article_ids': ', '.join(map(str, item['article_ids'])),
            'variation_count': len(item['variations']),
            'sample_variations': ' | '.join(item['variations'][:3])
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ Exported {len(rows)} across-article narratives to {output_path}")


def export_macro_arguments_csv(macro_arguments: list, output_path: str):
    """Export macro-arguments to CSV."""
    import csv

    if not macro_arguments:
        return

    rows = []
    for item in macro_arguments:
        rows.append({
            'topic': item['topic'],
            'narrative_count': item['narrative_count'],
            'article_count': item['article_count'],
            'article_ids': ', '.join(map(str, item['article_ids'][:20])) + (
                '...' if len(item['article_ids']) > 20 else '')
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ Exported {len(rows)} macro-arguments to {output_path}")


def process_articles(articles: list, article_ids: list, metadata_list: list, config: dict) -> tuple:
    """Run the narrative extraction pipeline with graph analysis."""

    print("=" * 60)
    print("NARRATIVE EXTRACTION & GRAPH ANALYSIS PIPELINE")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing agents...")
    agents = create_agent_pool(config)
    print(f"   Created {len(agents)} agents")

    print("\n2. Loading embedding model...")
    embedding_config = config.get('embeddings', {})
    embedding_matcher = EmbeddingMatcher(
        model_name=embedding_config.get('model_name', 'all-mpnet-base-v2'),
        device=embedding_config.get('device', None)
    )

    print("\n3. Initializing pipeline components...")
    article_extractor = ArticleNarrativeExtractor(
        agents=agents,
        embedding_model=embedding_matcher
    )

    narrative_graph = NarrativeGraph(
        embedding_model=embedding_matcher,
        similarity_threshold=config.get('pipeline', {}).get('graph', {}).get('similarity_threshold', 0.80)
    )

    # Process articles
    print(f"\n4. Processing {len(articles)} articles...")
    print("-" * 60)

    # Extract article-level narratives
    print("\n   Extracting article-level narratives...")
    article_results = article_extractor.batch_extract(articles, article_ids, metadata_list)

    # Build narrative graph
    print("\n   Building narrative graph...")
    narrative_graph.build_graph(article_results)

    # Get across-article narratives
    min_article_count = config.get('pipeline', {}).get('graph', {}).get('min_article_count', 3)
    across_article_narratives = narrative_graph.get_across_article_narratives(min_article_count)

    # Compile results
    total_narratives = sum(len(r['narratives']) for r in article_results)

    results = {
        'article_results': article_results,
        'across_article_narratives': across_article_narratives,
        'macro_arguments': [ma.to_dict() for ma in narrative_graph.macro_arguments],
        'graph_summary': narrative_graph.get_summary(),
        'summary': {
            'total_articles': len(articles),
            'total_narratives': total_narratives,
            'avg_narratives_per_article': total_narratives / len(articles) if articles else 0,
            'across_article_narrative_count': len(across_article_narratives),
            'macro_argument_count': len(narrative_graph.macro_arguments)
        }
    }

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(articles)} articles")
    print(f"Extracted: {total_narratives} total narratives")
    print(f"Identified: {len(across_article_narratives)} across-article narratives")
    print(f"Identified: {len(narrative_graph.macro_arguments)} macro-arguments")

    return results, narrative_graph


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run narrative extraction pipeline with graph analysis')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (JSON)')
    parser.add_argument('--output', type=str, default='data/output',
                        help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Configuration file')
    parser.add_argument('--query', type=str,
                        help='Optional: Run a query after processing (e.g., "actor:Trump")')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load articles
    print(f"Loading articles from {args.input}...")
    articles, article_ids, metadata_list = load_articles(args.input)
    print(f"Loaded {len(articles)} articles")

    # Process articles
    results, graph = process_articles(articles, article_ids, metadata_list, config)

    # Save results
    save_results(results, graph, args.output)

    # Run query if provided
    if args.query:
        print(f"\n{'=' * 60}")
        print(f"RUNNING QUERY: {args.query}")
        print('=' * 60)
        run_query(graph, args.query)

    print("\n✓ All done!")


def run_query(graph: NarrativeGraph, query_string: str):
    """Run a query on the graph."""
    # Simple query parser
    if query_string.startswith('actor:'):
        actor = query_string[6:]
        results = graph.query_by_actor(actor)
        print(f"\nFound {len(results)} narratives mentioning '{actor}':")
        for r in results[:10]:
            print(f"  - {r['narrative']} (in {r['article_count']} articles)")

    elif query_string.startswith('topic:'):
        topic = query_string[6:]
        results = graph.query_by_topic(topic)
        print(f"\nFound {len(results)} narratives about '{topic}':")
        for r in results[:10]:
            print(f"  - {r['narrative']} (in {r['article_count']} articles)")

    else:
        print(f"Unknown query format. Use 'actor:NAME' or 'topic:TOPIC'")


if __name__ == '__main__':
    main()