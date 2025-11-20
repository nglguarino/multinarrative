"""
Graph-based narrative analysis.

Creates a graph network of narratives to identify across-article patterns and macro-arguments.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json
from ..models.embeddings import EmbeddingMatcher


class NarrativeNode:
    """Represents a narrative node in the graph."""

    def __init__(self, narrative: str, node_id: int):
        self.narrative = narrative
        self.node_id = node_id
        self.article_ids: Set[int] = set()
        self.metadata: List[Dict[str, Any]] = []
        self.actors: Set[str] = set()
        self.topics: Set[str] = set()
        self.dates: List[datetime] = []
        self.places: Set[str] = set()
        self.similar_narratives: List[int] = []  # IDs of semantically similar narratives

    def add_article(self, article_id: int, metadata: Dict[str, Any]):
        """Add an article association to this narrative."""
        self.article_ids.add(article_id)
        self.metadata.append(metadata)

        # Extract structured information
        if 'actors' in metadata:
            self.actors.update(metadata['actors'])
        if 'topics' in metadata:
            self.topics.update(metadata['topics'])
        if 'date' in metadata:
            if isinstance(metadata['date'], datetime):
                self.dates.append(metadata['date'])
            elif isinstance(metadata['date'], str):
                try:
                    self.dates.append(datetime.fromisoformat(metadata['date']))
                except:
                    pass
        if 'places' in metadata:
            self.places.update(metadata['places'])

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'node_id': self.node_id,
            'narrative': self.narrative,
            'article_count': len(self.article_ids),
            'article_ids': sorted(list(self.article_ids)),
            'actors': sorted(list(self.actors)),
            'topics': sorted(list(self.topics)),
            'places': sorted(list(self.places)),
            'date_range': {
                'earliest': min(self.dates).isoformat() if self.dates else None,
                'latest': max(self.dates).isoformat() if self.dates else None
            },
            'similar_narrative_ids': self.similar_narratives
        }


class MacroArgument:
    """Represents a macro-argument (broad topic)."""

    def __init__(self, topic: str, narrative_ids: List[int]):
        self.topic = topic
        self.narrative_ids = narrative_ids
        self.article_ids: Set[int] = set()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'topic': self.topic,
            'narrative_count': len(self.narrative_ids),
            'narrative_ids': self.narrative_ids,
            'article_count': len(self.article_ids),
            'article_ids': sorted(list(self.article_ids))
        }


class NarrativeGraph:
    """
    Graph-based representation of narratives across articles.

    Enables querying by actors, topics, time, places, and composite queries.
    """

    def __init__(self, embedding_model: EmbeddingMatcher,
                 similarity_threshold: float = 0.80):
        """
        Initialize narrative graph.

        Args:
            embedding_model: Embedding model for semantic similarity
            similarity_threshold: Threshold for linking similar narratives
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        self.nodes: Dict[int, NarrativeNode] = {}  # node_id -> NarrativeNode
        self.narrative_to_node: Dict[str, int] = {}  # narrative text -> node_id
        self.macro_arguments: List[MacroArgument] = []

        self.next_node_id = 0

    def build_graph(self, article_results: List[Dict[str, Any]]):
        """
        Build graph from article extraction results.

        Args:
            article_results: List of results from ArticleNarrativeExtractor
        """
        print("Building narrative graph...")

        # Step 1: Create nodes for all unique narratives
        all_narratives = []
        narrative_to_articles = defaultdict(list)

        for result in article_results:
            article_id = result['article_id']
            metadata = result.get('metadata', {})

            for narrative in result['narratives']:
                all_narratives.append(narrative)
                narrative_to_articles[narrative].append((article_id, metadata))

        # Create nodes
        for narrative, article_data in narrative_to_articles.items():
            node_id = self.next_node_id
            self.next_node_id += 1

            node = NarrativeNode(narrative, node_id)
            for article_id, metadata in article_data:
                node.add_article(article_id, metadata)

            self.nodes[node_id] = node
            self.narrative_to_node[narrative] = node_id

        print(f"Created {len(self.nodes)} narrative nodes")

        # Step 2: Link similar narratives
        self._link_similar_narratives()

        # Step 3: Extract macro-arguments
        self._extract_macro_arguments()

        print(f"Identified {len(self.macro_arguments)} macro-arguments")

    def _link_similar_narratives(self):
        """Link semantically similar narratives in the graph."""
        print("Linking similar narratives...")

        narratives = [node.narrative for node in self.nodes.values()]
        if len(narratives) < 2:
            return

        # Compute similarity matrix
        similarity_matrix = self.embedding_model.batch_similarity(narratives)

        # Link similar narratives
        node_list = list(self.nodes.values())
        for i, node_i in enumerate(node_list):
            for j, node_j in enumerate(node_list):
                if i >= j:
                    continue

                similarity = similarity_matrix[i][j].item()
                if similarity >= self.similarity_threshold:
                    node_i.similar_narratives.append(node_j.node_id)
                    node_j.similar_narratives.append(node_i.node_id)

    def _extract_macro_arguments(self):
        """Extract macro-arguments by clustering related narratives."""
        print("Extracting macro-arguments...")

        # Use simple clustering based on semantic similarity
        visited = set()

        for node_id, node in self.nodes.items():
            if node_id in visited:
                continue

            # BFS to find connected component
            cluster = self._get_narrative_cluster(node_id, visited)

            if len(cluster) >= 3:  # Minimum 3 narratives for a macro-argument
                # Generate topic name using representative narratives
                topic = self._generate_topic_name(cluster)

                macro_arg = MacroArgument(topic, cluster)

                # Collect all article IDs
                for nid in cluster:
                    macro_arg.article_ids.update(self.nodes[nid].article_ids)

                self.macro_arguments.append(macro_arg)

    def _get_narrative_cluster(self, start_node_id: int, visited: Set[int]) -> List[int]:
        """Get cluster of related narratives using BFS."""
        cluster = []
        queue = [start_node_id]
        local_visited = set()

        while queue:
            node_id = queue.pop(0)
            if node_id in local_visited:
                continue

            local_visited.add(node_id)
            visited.add(node_id)
            cluster.append(node_id)

            # Add similar narratives to queue
            node = self.nodes[node_id]
            for similar_id in node.similar_narratives:
                if similar_id not in local_visited:
                    queue.append(similar_id)

        return cluster

    def _generate_topic_name(self, cluster: List[int]) -> str:
        """Generate a topic name for a cluster of narratives."""
        # Take most common words from narratives in cluster
        narratives = [self.nodes[nid].narrative for nid in cluster[:5]]

        # Simple approach: use the shortest narrative as representative
        narratives.sort(key=len)

        # Or combine common terms
        words = []
        for narrative in narratives:
            words.extend(narrative.split())

        # Count word frequency
        word_freq = defaultdict(int)
        for word in words:
            word_lower = word.lower().strip('.,!?')
            if len(word_lower) > 3:  # Skip short words
                word_freq[word_lower] += 1

        # Get top 3 words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        topic = " ".join([word for word, _ in top_words]).title()

        if not topic:
            topic = narratives[0][:50]  # Fallback

        return topic

    def get_across_article_narratives(self, min_article_count: int = 3) -> List[Dict[str, Any]]:
        """
        Get narratives that appear across multiple articles.

        Args:
            min_article_count: Minimum number of articles for across-article narrative

        Returns:
            List of across-article narratives with their clusters
        """
        across_article = []

        for node_id, node in self.nodes.items():
            # Get all nodes in this cluster
            cluster_nodes = [node_id] + node.similar_narratives
            cluster_nodes = list(set(cluster_nodes))

            # Collect all article IDs from cluster
            all_article_ids = set()
            for nid in cluster_nodes:
                all_article_ids.update(self.nodes[nid].article_ids)

            if len(all_article_ids) >= min_article_count:
                across_article.append({
                    'primary_narrative': node.narrative,
                    'node_id': node_id,
                    'variations': [self.nodes[nid].narrative for nid in node.similar_narratives],
                    'article_count': len(all_article_ids),
                    'article_ids': sorted(list(all_article_ids))
                })

        # Sort by article count
        across_article.sort(key=lambda x: x['article_count'], reverse=True)

        return across_article

    def query_by_actor(self, actor: str) -> List[Dict[str, Any]]:
        """Query narratives mentioning a specific actor."""
        results = []
        actor_lower = actor.lower()

        for node in self.nodes.values():
            if any(actor_lower in a.lower() for a in node.actors):
                results.append(node.to_dict())

        return results

    def query_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Query narratives related to a specific topic."""
        results = []
        topic_lower = topic.lower()

        for node in self.nodes.values():
            # Check if topic in narrative text or topics
            if (topic_lower in node.narrative.lower() or
                    any(topic_lower in t.lower() for t in node.topics)):
                results.append(node.to_dict())

        return results

    def query_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Query narratives within a date range."""
        results = []

        for node in self.nodes.values():
            if node.dates:
                node_dates = [d for d in node.dates if start_date <= d <= end_date]
                if node_dates:
                    results.append(node.to_dict())

        return results

    def query_by_place(self, place: str) -> List[Dict[str, Any]]:
        """Query narratives mentioning a specific place."""
        results = []
        place_lower = place.lower()

        for node in self.nodes.values():
            if any(place_lower in p.lower() for p in node.places):
                results.append(node.to_dict())

        return results

    def composite_query(self, actors: Optional[List[str]] = None,
                        topics: Optional[List[str]] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        places: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform composite query with multiple filters.

        Example: Find all narratives about "Trump" during "election period" mentioning "Pennsylvania"

        Args:
            actors: List of actor names to filter by
            topics: List of topics to filter by
            start_date: Start date for date range
            end_date: End date for date range
            places: List of places to filter by

        Returns:
            List of matching narratives
        """
        results = []

        for node in self.nodes.values():
            match = True

            # Check actors
            if actors:
                actor_match = any(
                    any(actor.lower() in a.lower() for a in node.actors)
                    for actor in actors
                )
                if not actor_match:
                    match = False

            # Check topics
            if topics and match:
                topic_match = any(
                    topic.lower() in node.narrative.lower() or
                    any(topic.lower() in t.lower() for t in node.topics)
                    for topic in topics
                )
                if not topic_match:
                    match = False

            # Check date range
            if start_date and end_date and match:
                if node.dates:
                    date_match = any(start_date <= d <= end_date for d in node.dates)
                    if not date_match:
                        match = False
                else:
                    match = False

            # Check places
            if places and match:
                place_match = any(
                    any(place.lower() in p.lower() for p in node.places)
                    for place in places
                )
                if not place_match:
                    match = False

            if match:
                results.append(node.to_dict())

        return results

    def export_graph(self, output_path: str):
        """Export graph to JSON file."""
        graph_data = {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'macro_arguments': [ma.to_dict() for ma in self.macro_arguments],
            'statistics': {
                'total_nodes': len(self.nodes),
                'total_macro_arguments': len(self.macro_arguments),
                'avg_articles_per_narrative': sum(len(n.article_ids) for n in self.nodes.values()) / len(
                    self.nodes) if self.nodes else 0
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        print(f"Exported graph to {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the graph."""
        return {
            'total_narratives': len(self.nodes),
            'total_macro_arguments': len(self.macro_arguments),
            'avg_articles_per_narrative': sum(len(n.article_ids) for n in self.nodes.values()) / len(
                self.nodes) if self.nodes else 0,
            'top_actors': self._get_top_entities('actors', 10),
            'top_topics': self._get_top_entities('topics', 10),
            'top_places': self._get_top_entities('places', 10)
        }

    def _get_top_entities(self, entity_type: str, limit: int) -> List[Tuple[str, int]]:
        """Get top entities by frequency."""
        entity_counts = defaultdict(int)

        for node in self.nodes.values():
            entities = getattr(node, entity_type, set())
            for entity in entities:
                entity_counts[entity] += 1

        return sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:limit]