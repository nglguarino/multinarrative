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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.agents import create_agent_pool
from src.models.embeddings import EmbeddingMatcher
from src.pipeline.paragraph_extractor import ParagraphNarrativeExtractor
from src.pipeline.article_aggregator import ArticleNarrativeAggregator
from src.pipeline.cross_article_analyzer import CrossArticleNarrativeAnalyzer


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


def load_articles(input_path: str) -> list:
    """
    Load articles from input file.
    
    Supports JSON format:
    - List of strings: ["article1", "article2", ...]
    - List of dicts: [{"id": 1, "text": "..."}, ...]
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            # List of strings
            return data, list(range(len(data)))
        elif all(isinstance(item, dict) for item in data):
            # List of dicts
            texts = [item.get('text', item.get('content', '')) for item in data]
            ids = [item.get('id', i) for i, item in enumerate(data)]
            return texts, ids
    
    raise ValueError("Invalid input format. Expected list of strings or list of dicts.")


def save_results(results: dict, output_dir: str, prefix: str = 'results'):
    """Save results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results as JSON
    json_path = os.path.join(output_dir, f'{prefix}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved results to {json_path}")
    
    # Save CSV export
    csv_path = os.path.join(output_dir, f'{prefix}.csv')
    export_to_csv(results, csv_path)
    
    # Save hierarchy JSON
    hierarchy_path = os.path.join(output_dir, f'{prefix}_hierarchy.json')
    export_hierarchy_json(results, hierarchy_path)


def export_to_csv(results: dict, output_path: str):
    """Export paragraph-level narratives to CSV."""
    import csv
    
    rows = []
    for article_result in results['article_results']:
        article_id = article_result['article_id']
        
        for para_result in article_result['paragraph_results']:
            para_idx = para_result['paragraph_index']
            para_text = para_result['paragraph_text']
            
            for narrative in para_result['narratives']:
                rows.append({
                    'article_id': article_id,
                    'paragraph_index': para_idx,
                    'paragraph_text': para_text[:200],
                    'narrative': narrative,
                    'confidence': para_result['confidence']
                })
    
    if rows:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Exported {len(rows)} paragraph narratives to {output_path}")


def export_hierarchy_json(results: dict, output_path: str):
    """Export complete hierarchy to JSON."""
    hierarchy = {
        'overarching': results['cross_article']['overarching_narratives'],
        'articles': []
    }
    
    for article_result in results['article_results']:
        article_entry = {
            'article_id': article_result['article_id'],
            'article_narratives': article_result['article_narratives'],
            'paragraphs': article_result['paragraph_results']
        }
        hierarchy['articles'].append(article_entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    print(f"✓ Exported hierarchy to {output_path}")


def process_articles(articles: list, article_ids: list, config: dict) -> dict:
    """Run the full narrative extraction pipeline."""
    
    print("="*60)
    print("NARRATIVE EXTRACTION PIPELINE")
    print("="*60)
    
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
    paragraph_extractor = ParagraphNarrativeExtractor(
        agents=agents,
        embedding_matcher=embedding_matcher,
        batch_size=config.get('pipeline', {}).get('paragraph', {}).get('batch_size', 5),
        max_workers=config.get('pipeline', {}).get('paragraph', {}).get('max_workers', 3)
    )
    
    article_aggregator = ArticleNarrativeAggregator(
        agents=agents,
        embedding_model=embedding_matcher
    )
    
    cross_article_analyzer = CrossArticleNarrativeAnalyzer(
        embedding_model=embedding_matcher,
        min_article_count=config.get('pipeline', {}).get('cross_article', {}).get('min_article_count', 2),
        similarity_threshold=config.get('pipeline', {}).get('cross_article', {}).get('similarity_threshold', 0.80)
    )
    
    # Process articles
    print(f"\n4. Processing {len(articles)} articles...")
    print("-" * 60)
    
    # Step 1: Extract paragraph-level narratives
    print("\n   Step 1: Extracting paragraph-level narratives...")
    paragraph_results = paragraph_extractor.batch_extract(articles, article_ids)
    
    # Step 2: Aggregate to article-level
    print("\n   Step 2: Aggregating to article-level narratives...")
    article_results = article_aggregator.batch_aggregate(paragraph_results)
    
    # Step 3: Identify cross-article narratives
    print("\n   Step 3: Identifying cross-article narratives...")
    cross_article_results = cross_article_analyzer.analyze(article_results)
    
    # Compile results
    total_paragraphs = sum(len(r['paragraph_results']) for r in article_results)
    total_para_narratives = sum(
        len(p['narratives']) 
        for r in article_results 
        for p in r['paragraph_results']
    )
    total_article_narratives = sum(len(r['article_narratives']) for r in article_results)
    
    results = {
        'article_results': article_results,
        'cross_article': cross_article_results,
        'summary': {
            'total_articles': len(articles),
            'total_paragraphs': total_paragraphs,
            'total_paragraph_narratives': total_para_narratives,
            'total_article_narratives': total_article_narratives,
            'overarching_narratives': len(cross_article_results['overarching_narratives'])
        }
    }
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Processed: {len(articles)} articles")
    print(f"Extracted: {total_para_narratives} paragraph narratives")
    print(f"Aggregated: {total_article_narratives} article narratives")
    print(f"Identified: {len(cross_article_results['overarching_narratives'])} overarching narratives")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run narrative extraction pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file (JSON)')
    parser.add_argument('--output', type=str, default='data/output',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load articles
    print(f"Loading articles from {args.input}...")
    articles, article_ids = load_articles(args.input)
    print(f"Loaded {len(articles)} articles")
    
    # Process articles
    results = process_articles(articles, article_ids, config)
    
    # Save results
    save_results(results, args.output)
    
    print("\n✓ All done!")


if __name__ == '__main__':
    main()
