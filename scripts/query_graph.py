"""
Query tool for narrative graph.

Usage:
    python scripts/query_graph.py --graph data/output/narrative_graph.json --actor Trump
    python scripts/query_graph.py --graph data/output/narrative_graph.json --actor Trump --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embeddings import EmbeddingMatcher
from src.pipeline.graph_analyzer import NarrativeGraph, NarrativeNode


def load_graph(graph_path: str, embedding_model: EmbeddingMatcher) -> NarrativeGraph:
    """Load graph from JSON file."""
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Reconstruct graph
    graph = NarrativeGraph(embedding_model)

    # Rebuild nodes
    for node_data in graph_data['nodes']:
        node = NarrativeNode(node_data['narrative'], node_data['node_id'])
        node.article_ids = set(node_data['article_ids'])
        node.actors = set(node_data['actors'])
        node.topics = set(node_data['topics'])
        node.places = set(node_data['places'])
        node.similar_narratives = node_data['similar_narrative_ids']

        graph.nodes[node.node_id] = node
        graph.narrative_to_node[node.narrative] = node.node_id

    graph.next_node_id = len(graph.nodes)

    print(f"Loaded graph with {len(graph.nodes)} narratives")

    return graph


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Query narrative graph')
    parser.add_argument('--graph', type=str, required=True,
                       help='Path to narrative_graph.json')
    parser.add_argument('--actor', type=str,
                       help='Query by actor name')
    parser.add_argument('--topic', type=str,
                       help='Query by topic')
    parser.add_argument('--place', type=str,
                       help='Query by place')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str,
                       help='Optional: Save results to JSON file')

    args = parser.parse_args()

    # Load embedding model
    print("Loading embedding model...")
    embedding_model = EmbeddingMatcher()

    # Load graph
    print(f"Loading graph from {args.graph}...")
    graph = load_graph(args.graph, embedding_model)

    # Build query parameters
    actors = [args.actor] if args.actor else None
    topics = [args.topic] if args.topic else None
    places = [args.place] if args.place else None
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else None

    # Run query
    print("\nRunning query...")
    if actors or topics or places or (start_date and end_date):
        results = graph.composite_query(
            actors=actors,
            topics=topics,
            start_date=start_date,
            end_date=end_date,
            places=places
        )
    elif args.actor:
        results = graph.query_by_actor(args.actor)
    elif args.topic:
        results = graph.query_by_topic(args.topic)
    elif args.place:
        results = graph.query_by_place(args.place)
    else:
        print("Error: Please specify at least one query parameter")
        return

    # Display results
    print(f"\n{'='*60}")
    print(f"QUERY RESULTS: {len(results)} narratives found")
    print('='*60)

    for i, result in enumerate(results[:20], 1):
        print(f"\n{i}. {result['narrative']}")
        print(f"   Articles: {result['article_count']}")
        print(f"   IDs: {result['article_ids'][:10]}{'...' if len(result['article_ids']) > 10 else ''}")
        if result['actors']:
            print(f"   Actors: {', '.join(list(result['actors'])[:5])}")
        if result['topics']:
            print(f"   Topics: {', '.join(list(result['topics'])[:5])}")

    if len(results) > 20:
        print(f"\n... and {len(results) - 20} more results")

    # Save if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved results to {args.output}")


if __name__ == '__main__':
    main()