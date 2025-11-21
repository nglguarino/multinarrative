"""
Parallel metadata extraction for faster processing.

Uses multi-threading to process multiple articles simultaneously.
"""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .metadata_extractor import AdvancedMetadataExtractor


class ParallelMetadataExtractor:
    """Extract metadata using multiple threads for faster processing."""

    def __init__(self, use_gpu: bool = False, num_workers: int = 4):
        """
        Initialize parallel extractor.

        Args:
            use_gpu: Whether to use GPU
            num_workers: Number of parallel workers (threads)
        """
        self.num_workers = num_workers
        self.use_gpu = use_gpu

        # Create a single extractor (will be reused across threads)
        # Note: For GPU, we use single extractor; for CPU, multiple extractors work better
        print(f"Initializing parallel extractor with {num_workers} workers...")
        if use_gpu:
            # Single extractor for GPU
            self.extractor = AdvancedMetadataExtractor(use_gpu=True)
            self.extractors = [self.extractor] * num_workers
        else:
            # Multiple extractors for CPU
            self.extractors = [
                AdvancedMetadataExtractor(use_gpu=False)
                for _ in range(num_workers)
            ]

    def batch_extract(self, articles: List[str], filenames: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract metadata from multiple articles in parallel.

        Args:
            articles: List of article texts
            filenames: Optional list of filenames

        Returns:
            List of metadata dictionaries
        """
        if filenames is None:
            filenames = [None] * len(articles)

        total = len(articles)
        print(f"Extracting metadata from {total} articles using {self.num_workers} workers...")

        # For GPU, use the optimized batch extraction
        if self.use_gpu:
            return self.extractor.batch_extract(articles, filenames)

        # For CPU, use parallel processing
        metadata_list = [None] * total
        completed = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Create worker-task assignments
            futures = {}
            for idx, (article, filename) in enumerate(zip(articles, filenames)):
                # Round-robin assignment to workers
                worker_idx = idx % self.num_workers
                extractor = self.extractors[worker_idx]

                future = executor.submit(self._extract_single, extractor, article, filename)
                futures[future] = idx

            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    metadata = future.result()
                    metadata_list[idx] = metadata
                    completed += 1

                    if completed % 50 == 0 or completed == total:
                        print(f"  Progress: {completed}/{total}")
                except Exception as e:
                    print(f"  Error processing article {idx}: {e}")
                    metadata_list[idx] = {}

        print("✓ Metadata extraction complete\n")
        return metadata_list

    def _extract_single(self, extractor: AdvancedMetadataExtractor,
                        article: str, filename: Optional[str]) -> Dict[str, Any]:
        """Extract metadata from single article."""
        return extractor.extract_metadata(article, filename)