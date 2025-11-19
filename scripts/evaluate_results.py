"""
Evaluate and analyze narrative extraction results.

Usage:
    python scripts/evaluate_results.py --results data/output/results.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_results(results: dict):
    """
    Analyze and display results from the narrative extraction pipeline.
    
    Args:
        results: Results dictionary from run_extraction.py
    """
    print("="*60)
    print("NARRATIVE EXTRACTION ANALYSIS")
    print("="*60)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Articles processed: {results['summary']['total_articles']}")
    print(f"  Total paragraphs: {results['summary']['total_paragraphs']}")
    print(f"  Paragraph narratives: {results['summary']['total_paragraph_narratives']}")
    print(f"  Article narratives: {results['summary']['total_article_narratives']}")
    print(f"  Overarching narratives: {results['summary']['overarching_narratives']}")
    
    # Paragraph-level statistics
    print(f"\n📄 PARAGRAPH-LEVEL ANALYSIS:")
    if results['summary']['total_paragraphs'] > 0:
        avg_narratives_per_para = (
            results['summary']['total_paragraph_narratives'] / 
            results['summary']['total_paragraphs']
        )
        print(f"  Avg narratives per paragraph: {avg_narratives_per_para:.2f}")
    
    para_with_narratives = sum(
        1 for r in results['article_results']
        for p in r['paragraph_results'] 
        if p['narratives']
    )
    print(f"  Paragraphs with narratives: {para_with_narratives}/{results['summary']['total_paragraphs']}")
    
    # Confidence distribution
    confidences = [
        p['confidence'] 
        for r in results['article_results']
        for p in r['paragraph_results']
    ]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"  Average confidence: {avg_confidence:.2f}")
    
    # Article-level statistics
    print(f"\n📰 ARTICLE-LEVEL ANALYSIS:")
    if results['summary']['total_articles'] > 0:
        avg_article_narratives = (
            results['summary']['total_article_narratives'] / 
            results['summary']['total_articles']
        )
        print(f"  Avg narratives per article: {avg_article_narratives:.2f}")
    
    # Distribution of narratives per article
    narrative_counts = [
        len(r['article_narratives']) 
        for r in results['article_results']
    ]
    print(f"  Min narratives in article: {min(narrative_counts) if narrative_counts else 0}")
    print(f"  Max narratives in article: {max(narrative_counts) if narrative_counts else 0}")
    
    # Cross-article analysis
    print(f"\n🌍 OVERARCHING NARRATIVES (Top 10):")
    for i, narr in enumerate(results['cross_article']['overarching_narratives'][:10], 1):
        print(f"\n  {i}. {narr['narrative']}")
        print(f"     └─ Appears in {narr['article_count']} articles")
        print(f"     └─ Article IDs: {narr['article_ids'][:5]}{'...' if len(narr['article_ids']) > 5 else ''}")
        if narr.get('variations'):
            print(f"     └─ Variations: {len(narr['variations'])}")
    
    # Sample hierarchy
    print(f"\n🏗️  HIERARCHY EXAMPLE (Article 0):")
    if results['article_results']:
        article_0 = results['article_results'][0]
        
        print(f"\n  📰 ARTICLE-LEVEL NARRATIVES ({len(article_0['article_narratives'])}):") 
        for i, narr in enumerate(article_0['article_narratives'][:5], 1):
            print(f"    {i}. {narr}")
        
        print(f"\n  📄 PARAGRAPH-LEVEL NARRATIVES (first 3 paragraphs):")
        for para in article_0['paragraph_results'][:3]:
            if para['narratives']:
                print(f"\n    Paragraph {para['paragraph_index']} ({len(para['narratives'])} narratives):")
                print(f"    Text: {para['paragraph_text'][:100]}...")
                for j, narr in enumerate(para['narratives'][:3], 1):
                    print(f"      {j}. {narr}")


def compute_coverage_stats(results: dict) -> dict:
    """Compute coverage statistics."""
    articles_with_narratives = sum(
        1 for r in results['article_results']
        if r['article_narratives']
    )
    
    paragraphs_with_narratives = sum(
        1 for r in results['article_results']
        for p in r['paragraph_results']
        if p['narratives']
    )
    
    return {
        'article_coverage': articles_with_narratives / max(results['summary']['total_articles'], 1),
        'paragraph_coverage': paragraphs_with_narratives / max(results['summary']['total_paragraphs'], 1)
    }


def export_narrative_frequency(results: dict, output_path: str):
    """Export narrative frequency analysis."""
    # Count narrative occurrences
    narrative_counter = Counter()
    
    for article in results['article_results']:
        for narrative in article['article_narratives']:
            narrative_counter[narrative] += 1
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Narrative,Frequency\n")
        for narrative, count in narrative_counter.most_common():
            f.write(f'"{narrative}",{count}\n')
    
    print(f"✓ Exported narrative frequency to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze narrative extraction results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--export-freq', type=str,
                       help='Optional: Export narrative frequency to CSV')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Analyze
    analyze_results(results)
    
    # Compute coverage
    print(f"\n📈 COVERAGE STATISTICS:")
    coverage = compute_coverage_stats(results)
    print(f"  Article coverage: {coverage['article_coverage']*100:.1f}%")
    print(f"  Paragraph coverage: {coverage['paragraph_coverage']*100:.1f}%")
    
    # Export frequency if requested
    if args.export_freq:
        export_narrative_frequency(results, args.export_freq)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
