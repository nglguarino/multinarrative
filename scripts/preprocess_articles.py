"""
Preprocess raw .txt files into structured JSON with metadata.

Usage:
    # Process single directory
    python scripts/preprocess_articles.py --input data/input/raw_articles/ --output data/input/articles.json

    # Process with specific config
    python scripts/preprocess_articles.py --input data/input/raw_articles/ --output data/input/articles.json --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_loader import ArticleLoader
from src.utils.metadata_extractor import AdvancedMetadataExtractor
import yaml
import os


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    for key in config.get('api_keys', {}):
        value = config['api_keys'][key]
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            config['api_keys'][key] = os.getenv(env_var)

    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Preprocess raw .txt articles into structured JSON with metadata'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing .txt files OR single .txt file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Configuration file')
    parser.add_argument('--pattern', type=str, default='*.txt',
                        help='File pattern to match (default: *.txt)')

    args = parser.parse_args()

    print("=" * 60)
    print("ARTICLE PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load config
    config = load_config(args.config)

    # Load articles
    print(f"\n1. Loading articles from {args.input}...")

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        text, filename = ArticleLoader.load_from_txt_file(str(input_path))
        articles = [text]
        filenames = [filename]
        article_ids = [0]
    elif input_path.is_dir():
        # Directory
        articles, filenames, article_ids = ArticleLoader.load_from_directory(
            str(input_path),
            args.pattern
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return

    print(f"   Loaded {len(articles)} articles")

    # Initialize metadata extractor
    print("\n2. Initializing NLP/ML metadata extraction...")
    use_gpu = config.get('metadata', {}).get('use_gpu', False)
    metadata_extractor = AdvancedMetadataExtractor(use_gpu=use_gpu)

    # Extract metadata
    print("\n3. Extracting metadata...")
    metadata_list = metadata_extractor.batch_extract(articles, filenames)

    # Display sample
    print("\n   Sample extracted metadata:")
    for i in range(min(3, len(metadata_list))):
        print(f"\n   Article {i}: {filenames[i]}")
        print(f"   Title: {metadata_list[i].get('title', 'N/A')}")
        print(f"   Date: {metadata_list[i].get('date', 'N/A')}")
        print(f"   Actors: {', '.join(metadata_list[i].get('actors', [])[:3])}")
        print(f"   Topics: {', '.join(metadata_list[i].get('topics', [])[:3])}")
        print(f"   Places: {', '.join(metadata_list[i].get('places', [])[:3])}")

    # Save to JSON
    print(f"\n4. Saving to {args.output}...")
    ArticleLoader.save_to_json(articles, article_ids, metadata_list, args.output)

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(articles)} articles")
    print(f"Output: {args.output}")

    # Statistics
    dates_found = sum(1 for m in metadata_list if m.get('date'))
    actors_found = sum(1 for m in metadata_list if m.get('actors'))
    topics_found = sum(1 for m in metadata_list if m.get('topics'))
    places_found = sum(1 for m in metadata_list if m.get('places'))

    print(f"\nMetadata Statistics:")
    print(f"  Articles with dates: {dates_found}/{len(articles)} ({dates_found / len(articles) * 100:.1f}%)")
    print(f"  Articles with actors: {actors_found}/{len(articles)} ({actors_found / len(articles) * 100:.1f}%)")
    print(f"  Articles with topics: {topics_found}/{len(articles)} ({topics_found / len(articles) * 100:.1f}%)")
    print(f"  Articles with places: {places_found}/{len(articles)} ({places_found / len(articles) * 100:.1f}%)")

    print("\n✓ Ready for narrative extraction!")
    print(f"   Next step: python scripts/run_extraction.py --input {args.output} --output data/output/")


if __name__ == '__main__':
    main()