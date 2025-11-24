"""
Interactive graph visualization for narrative networks.

Uses Plotly for interactive 3D/2D network visualization with clustering.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import networkx as nx


class NarrativeGraphVisualizer:
    """
    Create interactive visualizations of narrative graphs.

    Features:
    - 2D/3D network layouts
    - Node clustering by semantic similarity
    - Interactive hover information
    - Color-coded by topic/actor/macro-argument
    - Filterable views
    """

    def __init__(self, graph_path: str = None, graph_data: dict = None,
                 embedding_model=None):
        """
        Initialize visualizer.

        Args:
            graph_path: Path to narrative_graph.json
            graph_data: Alternatively, provide graph data directly
            embedding_model: Optional embedding model for better clustering
        """
        if graph_path:
            with open(graph_path, 'r', encoding='utf-8') as f:
                self.graph_data = json.load(f)
        elif graph_data:
            self.graph_data = graph_data
        else:
            raise ValueError("Must provide either graph_path or graph_data")

        self.embedding_model = embedding_model
        self.nodes = self.graph_data['nodes']
        self.macro_arguments = self.graph_data.get('macro_arguments', [])

        # Build NetworkX graph for layout algorithms
        self.nx_graph = self._build_networkx_graph()

        # Compute embeddings for better clustering
        self.embeddings = None
        if embedding_model:
            narratives = [node['narrative'] for node in self.nodes]
            embeddings_tensor = embedding_model.model.encode(
                narratives,
                convert_to_tensor=True
            )
            self.embeddings = embeddings_tensor.cpu().numpy()

    def _build_networkx_graph(self) -> nx.Graph:
        """Build NetworkX graph from node data."""
        G = nx.Graph()

        # Add nodes
        for node in self.nodes:
            G.add_node(
                node['node_id'],
                narrative=node['narrative'],
                article_count=node['article_count'],
                actors=node['actors'],
                topics=node['topics']
            )

        # Add edges based on similar narratives
        for node in self.nodes:
            node_id = node['node_id']
            for similar_id in node['similar_narrative_ids']:
                if similar_id in G.nodes:
                    G.add_edge(node_id, similar_id)

        return G

    def create_2d_visualization(self,
                                color_by: str = 'cluster',
                                min_article_count: int = 1,
                                layout: str = 'spring',
                                n_clusters: int = 10) -> go.Figure:
        """
        Create 2D interactive network visualization.

        Args:
            color_by: Color nodes by 'cluster', 'topic', 'actor', or 'article_count'
            min_article_count: Minimum articles to show node
            layout: 'spring', 'kamada_kawai', or 'circular'
            n_clusters: Number of clusters for color coding

        Returns:
            Plotly figure object
        """
        # Filter nodes
        filtered_nodes = [
            node for node in self.nodes
            if node['article_count'] >= min_article_count
        ]

        if not filtered_nodes:
            raise ValueError("No nodes match filter criteria")

        # Get node IDs
        node_ids = [node['node_id'] for node in filtered_nodes]

        # Create subgraph
        subgraph = self.nx_graph.subgraph(node_ids)

        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subgraph)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph)

        # Compute colors
        colors, color_labels = self._compute_colors(
            filtered_nodes, color_by, n_clusters
        )

        # Extract coordinates
        x_nodes = [pos[node_id][0] for node_id in node_ids]
        y_nodes = [pos[node_id][1] for node_id in node_ids]

        # Create edges
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )

        # Create node trace
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True if color_by == 'article_count' else False,
                colorscale='Viridis' if color_by == 'article_count' else None,
                color=colors,
                size=[min(30, 5 + node['article_count'] * 2) for node in filtered_nodes],
                colorbar=dict(
                    thickness=15,
                    title=color_labels if color_by == 'article_count' else '',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            ),
            text=[self._create_hover_text(node) for node in filtered_nodes],
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=f'Narrative Graph Network ({len(filtered_nodes)} narratives, colored by {color_by})',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=800
        )

        return fig

    def create_3d_visualization(self,
                                color_by: str = 'cluster',
                                min_article_count: int = 1,
                                n_clusters: int = 10) -> go.Figure:
        """
        Create 3D interactive network visualization.

        Args:
            color_by: Color nodes by 'cluster', 'topic', 'actor', or 'article_count'
            min_article_count: Minimum articles to show node
            n_clusters: Number of clusters for color coding

        Returns:
            Plotly figure object
        """
        # Filter nodes
        filtered_nodes = [
            node for node in self.nodes
            if node['article_count'] >= min_article_count
        ]

        if not filtered_nodes:
            raise ValueError("No nodes match filter criteria")

        # Get node IDs
        node_ids = [node['node_id'] for node in filtered_nodes]

        # Create subgraph
        subgraph = self.nx_graph.subgraph(node_ids)

        # Compute 3D layout using spring layout
        pos = nx.spring_layout(subgraph, dim=3, k=2, iterations=50, seed=42)

        # Compute colors
        colors, color_labels = self._compute_colors(
            filtered_nodes, color_by, n_clusters
        )

        # Extract coordinates
        x_nodes = [pos[node_id][0] for node_id in node_ids]
        y_nodes = [pos[node_id][1] for node_id in node_ids]
        z_nodes = [pos[node_id][2] for node_id in node_ids]

        # Create edges
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in subgraph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        # Create edge trace
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )

        # Create node trace
        node_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=colors,
                size=[min(20, 3 + node['article_count']) for node in filtered_nodes],
                colorscale='Viridis' if color_by == 'article_count' else None,
                showscale=True if color_by == 'article_count' else False,
                colorbar=dict(
                    thickness=15,
                    title=color_labels if color_by == 'article_count' else '',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=1, color='white')
            ),
            text=[self._create_hover_text(node) for node in filtered_nodes],
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=f'3D Narrative Graph Network ({len(filtered_nodes)} narratives)',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='white'
            ),
            height=800
        )

        return fig

    def create_cluster_view(self, n_clusters: int = 10,
                            perplexity: int = 30) -> go.Figure:
        """
        Create t-SNE cluster visualization.

        Args:
            n_clusters: Number of clusters for KMeans
            perplexity: t-SNE perplexity parameter

        Returns:
            Plotly figure object
        """
        if self.embeddings is None:
            raise ValueError("Embeddings required for cluster view. Provide embedding_model.")

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(self.nodes) - 1),
                    random_state=42)
        tsne_coords = tsne.fit_transform(self.embeddings)

        # Run KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.embeddings)

        # Create figure with clusters
        fig = go.Figure()

        # Color palette
        colors = [
                     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                 ] * 10  # Repeat if needed

        # Add trace for each cluster
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            cluster_nodes = [node for i, node in enumerate(self.nodes) if mask[i]]

            fig.add_trace(go.Scatter(
                x=tsne_coords[mask, 0],
                y=tsne_coords[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    color=colors[cluster_id],
                    size=[min(30, 5 + node['article_count'] * 2) for node in cluster_nodes],
                    line=dict(width=1, color='white')
                ),
                text=[self._create_hover_text(node) for node in cluster_nodes],
                hoverinfo='text'
            ))

        fig.update_layout(
            title=f't-SNE Cluster Visualization ({n_clusters} clusters)',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            hovermode='closest',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def create_macro_argument_view(self) -> go.Figure:
        """
        Create visualization of macro-arguments.

        Returns:
            Plotly figure object
        """
        if not self.macro_arguments:
            raise ValueError("No macro-arguments found in graph data")

        # Sort by article count
        sorted_macros = sorted(
            self.macro_arguments,
            key=lambda x: x['article_count'],
            reverse=True
        )[:20]  # Top 20

        topics = [m['topic'] for m in sorted_macros]
        narrative_counts = [m['narrative_count'] for m in sorted_macros]
        article_counts = [m['article_count'] for m in sorted_macros]

        # Create figure with secondary axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=topics,
                y=narrative_counts,
                name="Narrative Count",
                marker_color='lightblue'
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Bar(
                x=topics,
                y=article_counts,
                name="Article Count",
                marker_color='coral'
            ),
            secondary_y=True,
        )

        fig.update_xaxes(title_text="Macro-Argument Topic", tickangle=-45)
        fig.update_yaxes(title_text="Number of Narratives", secondary_y=False)
        fig.update_yaxes(title_text="Number of Articles", secondary_y=True)

        fig.update_layout(
            title="Top 20 Macro-Arguments by Scope",
            hovermode='x unified',
            height=600,
            showlegend=True
        )

        return fig

    def create_topic_distribution(self, top_n: int = 15) -> go.Figure:
        """
        Create topic distribution visualization.

        Args:
            top_n: Number of top topics to show

        Returns:
            Plotly figure object
        """
        # Count topics across all nodes
        topic_counts = {}
        for node in self.nodes:
            for topic in node['topics']:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Sort and get top N
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

        topics = [t[0] for t in sorted_topics]
        counts = [t[1] for t in sorted_topics]

        fig = go.Figure(data=[
            go.Bar(
                x=counts,
                y=topics,
                orientation='h',
                marker=dict(
                    color=counts,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Narrative Count")
                ),
                text=counts,
                textposition='auto'
            )
        ])

        fig.update_layout(
            title=f'Top {top_n} Topics by Narrative Count',
            xaxis_title='Number of Narratives',
            yaxis_title='Topic',
            height=600,
            showlegend=False
        )

        return fig

    def create_actor_network(self, min_narratives: int = 10) -> go.Figure:
        """
        Create actor co-occurrence network.

        Args:
            min_narratives: Minimum narratives for actor to appear

        Returns:
            Plotly figure object
        """
        # Count actor co-occurrences
        actor_counts = {}
        actor_cooccurrence = {}

        for node in self.nodes:
            actors = node['actors']
            for actor in actors:
                actor_counts[actor] = actor_counts.get(actor, 0) + 1

            # Co-occurrences
            for i, actor1 in enumerate(actors):
                for actor2 in actors[i + 1:]:
                    pair = tuple(sorted([actor1, actor2]))
                    actor_cooccurrence[pair] = actor_cooccurrence.get(pair, 0) + 1

        # Filter actors
        top_actors = {
            actor for actor, count in actor_counts.items()
            if count >= min_narratives
        }

        # Build NetworkX graph
        G = nx.Graph()
        for actor in top_actors:
            G.add_node(actor, count=actor_counts[actor])

        for (actor1, actor2), count in actor_cooccurrence.items():
            if actor1 in top_actors and actor2 in top_actors and count >= 3:
                G.add_edge(actor1, actor2, weight=count)

        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Create edges
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])

        # Normalize edge weights for visualization
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 5 for w in edge_weights]

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )

        # Create node trace
        x_nodes = [pos[actor][0] for actor in G.nodes()]
        y_nodes = [pos[actor][1] for actor in G.nodes()]
        node_sizes = [G.nodes[actor]['count'] * 2 for actor in G.nodes()]

        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(
                size=node_sizes,
                color='lightblue',
                line=dict(width=2, color='white')
            ),
            hovertext=[f"{actor}: {G.nodes[actor]['count']} narratives"
                       for actor in G.nodes()],
            showlegend=False
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=f'Actor Co-occurrence Network (min {min_narratives} narratives)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )

        return fig

    def _compute_colors(self, nodes: List[Dict], color_by: str,
                        n_clusters: int) -> Tuple[List, str]:
        """Compute colors for nodes based on coloring scheme."""
        if color_by == 'article_count':
            colors = [node['article_count'] for node in nodes]
            label = "Article Count"

        elif color_by == 'cluster' and self.embeddings is not None:
            # Use KMeans clustering on embeddings
            node_indices = [node['node_id'] for node in nodes]
            node_embeddings = self.embeddings[node_indices]

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(node_embeddings)
            colors = clusters.tolist()
            label = "Cluster"

        elif color_by == 'topic':
            # Color by primary topic
            topic_to_id = {}
            colors = []
            for node in nodes:
                if node['topics']:
                    topic = node['topics'][0]
                    if topic not in topic_to_id:
                        topic_to_id[topic] = len(topic_to_id)
                    colors.append(topic_to_id[topic])
                else:
                    colors.append(0)
            label = "Primary Topic"

        else:
            # Default: all same color
            colors = ['blue'] * len(nodes)
            label = ""

        return colors, label

    def _create_hover_text(self, node: Dict) -> str:
        """Create hover text for a node."""
        text = f"<b>{node['narrative']}</b><br>"
        text += f"Articles: {node['article_count']}<br>"

        if node['actors']:
            actors_str = ', '.join(node['actors'][:5])
            if len(node['actors']) > 5:
                actors_str += '...'
            text += f"Actors: {actors_str}<br>"

        if node['topics']:
            topics_str = ', '.join(node['topics'][:3])
            text += f"Topics: {topics_str}<br>"

        return text

    def save_all_visualizations(self, output_dir: str):
        """
        Generate and save all visualizations.

        Args:
            output_dir: Directory to save HTML files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Generating visualizations...")

        # 2D Network
        print("  Creating 2D network...")
        fig_2d = self.create_2d_visualization(color_by='cluster')
        fig_2d.write_html(os.path.join(output_dir, 'network_2d.html'))

        # 3D Network
        print("  Creating 3D network...")
        fig_3d = self.create_3d_visualization(color_by='cluster')
        fig_3d.write_html(os.path.join(output_dir, 'network_3d.html'))

        # Cluster view (if embeddings available)
        if self.embeddings is not None:
            print("  Creating cluster view...")
            fig_cluster = self.create_cluster_view()
            fig_cluster.write_html(os.path.join(output_dir, 'clusters_tsne.html'))

        # Macro-arguments
        if self.macro_arguments:
            print("  Creating macro-argument view...")
            fig_macro = self.create_macro_argument_view()
            fig_macro.write_html(os.path.join(output_dir, 'macro_arguments.html'))

        # Topic distribution
        print("  Creating topic distribution...")
        fig_topics = self.create_topic_distribution()
        fig_topics.write_html(os.path.join(output_dir, 'topic_distribution.html'))

        # Actor network
        print("  Creating actor network...")
        fig_actors = self.create_actor_network()
        fig_actors.write_html(os.path.join(output_dir, 'actor_network.html'))

        print(f"\n✓ All visualizations saved to {output_dir}/")
        print(f"  Open the .html files in a web browser to explore interactively.")