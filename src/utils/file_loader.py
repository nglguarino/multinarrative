"""
File loading utilities for various input formats.

Handles loading articles from .txt files, directories, and structured formats.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


class ArticleLoader:
    """Load articles from various file formats."""

    @staticmethod
    def load_from_txt_file(filepath: str) -> Tuple[str, str]:
        """
        Load a single .txt file.

        Args:
            filepath: Path to .txt file

        Returns:
            Tuple of (article_text, filename)
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        filename = os.path.basename(filepath)
        return text, filename

    @staticmethod
    def load_from_directory(directory: str, pattern: str = "*.txt") -> Tuple[List[str], List[str], List[int]]:
        """
        Load all .txt files from a directory.

        Args:
            directory: Path to directory containing .txt files
            pattern: File pattern to match (default: "*.txt")

        Returns:
            Tuple of (articles, filenames, article_ids)
        """
        directory_path = Path(directory)

        # Find all matching files
        files = sorted(directory_path.glob(pattern))

        if not files:
            raise ValueError(f"No files matching '{pattern}' found in {directory}")

        articles = []
        filenames = []
        article_ids = []

        print(f"Loading {len(files)} files from {directory}...")

        for i, filepath in enumerate(files):
            try:
                text, filename = ArticleLoader.load_from_txt_file(str(filepath))
                articles.append(text)
                filenames.append(filename)
                article_ids.append(i)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

        print(f"✓ Loaded {len(articles)} articles")

        return articles, filenames, article_ids

    @staticmethod
    def load_from_json(filepath: str) -> Tuple[List[str], List[int], List[Dict[str, Any]], Optional[List[str]]]:
        """
        Load articles from JSON file.

        Supports formats:
        - List of strings: ["article1", "article2", ...]
        - List of dicts: [{"id": 1, "text": "...", "metadata": {...}}, ...]

        Args:
            filepath: Path to JSON file

        Returns:
            Tuple of (articles, article_ids, metadata_list, filenames)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                # List of strings
                return data, list(range(len(data))), [{}] * len(data), None

            elif all(isinstance(item, dict) for item in data):
                # List of dicts
                articles = []
                article_ids = []
                metadata_list = []
                filenames = []

                for i, item in enumerate(data):
                    articles.append(item.get('text', item.get('content', '')))
                    article_ids.append(item.get('id', i))
                    metadata_list.append(item.get('metadata', {}))
                    filenames.append(item.get('filename', f"article_{i}.txt"))

                return articles, article_ids, metadata_list, filenames

        raise ValueError("Invalid JSON format. Expected list of strings or list of dicts.")

    @staticmethod
    def save_to_json(articles: List[str], article_ids: List[int],
                     metadata_list: List[Dict[str, Any]], output_path: str):
        """
        Save articles with metadata to JSON file.

        Args:
            articles: List of article texts
            article_ids: List of article IDs
            metadata_list: List of metadata dictionaries
            output_path: Path to save JSON file
        """
        data = []

        for article_id, text, metadata in zip(article_ids, articles, metadata_list):
            data.append({
                'id': article_id,
                'text': text,
                'metadata': metadata
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(data)} articles with metadata to {output_path}")