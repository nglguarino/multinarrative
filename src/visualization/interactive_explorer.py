"""
Enhanced interactive explorer with filtering and search.

Provides widgets-based interface for exploring narrative graphs.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go


class InteractiveNarrativeExplorer:
    """
    Interactive explorer for narrative graphs with filtering and search.

    Features:
    - Filter by topic, actor, article count
    - Search narratives
    - Dynamic visualization updates
    - Export filtered results
    """

    def __init__(self, visualizer, graph):
        """
        Initialize explorer.

        Args:
            visualizer: NarrativeGraphVisualizer instance
            graph: NarrativeGraph instance
        """
        self.visualizer = visualizer
        self.graph = graph
        self.filtered_nodes = visualizer.nodes.copy()

        # Extract unique values
        self.all_topics = self._extract_unique_topics()
        self.all_actors = self._extract_unique_actors()

        # Create widgets
        self._create_widgets()

    def _extract_unique_topics(self) -> List[str]:
        """Extract all unique topics."""
        topics = set()
        for node in self.visualizer.nodes:
            topics.update(node['topics'])
        return sorted(list(topics))

    def _extract_unique_actors(self) -> List[str]:
        """Extract all unique actors."""
        actors = set()
        for node in self.visualizer.nodes:
            actors.update(node['actors'])
        return sorted(list(actors))

    def _create_widgets(self):
        """Create UI widgets."""
        # Search box
        self.search_box = widgets.Text(
            placeholder='Search narratives...',
            description='Search:',
            style={'description_width': '100px'}
        )

        # Topic filter
        self.topic_filter = widgets.SelectMultiple(
            options=['All'] + self.all_topics,
            value=['All'],
            description='Topics:',
            rows=5,
            style={'description_width': '100px'}
        )

        # Actor filter
        self.actor_filter = widgets.SelectMultiple(
            options=['All'] + self.all_actors[:50],  # Top 50 actors
            value=['All'],
            description='Actors:',
            rows=5,
            style={'description_width': '100px'}
        )

        # Article count slider
        self.article_count_slider = widgets.IntSlider(
            value=1,
            min=1,
            max=max(node['article_count'] for node in self.visualizer.nodes),
            step=1,
            description='Min Articles:',
            style={'description_width': '100px'}
        )

        # Visualization type
        self.viz_type = widgets.Dropdown(
            options=['2D Network', '3D Network', 't-SNE Clusters', 'Topic Distribution'],
            value='2D Network',
            description='View:',
            style={'description_width': '100px'}
        )

        # Color by
        self.color_by = widgets.Dropdown(
            options=['cluster', 'topic', 'article_count'],
            value='cluster',
            description='Color by:',
            style={'description_width': '100px'}
        )

        # Update button
        self.update_button = widgets.Button(
            description='Update Visualization',
            button_style='primary',
            icon='refresh'
        )

        # Export button
        self.export_button = widgets.Button(
            description='Export Filtered Data',
            button_style='success',
            icon='download'
        )

        # Results display
        self.results_text = widgets.HTML(
            value="<b>Total narratives:</b> " + str(len(self.visualizer.nodes))
        )

        # Attach callbacks
        self.update_button.on_click(self._on_update_click)
        self.export_button.on_click(self._on_export_click)

    def display(self):
        """Display the interactive explorer interface."""
        # Layout
        filters = widgets.VBox([
            widgets.HTML("<h3>Filters</h3>"),
            self.search_box,
            self.topic_filter,
            self.actor_filter,
            self.article_count_slider,
            widgets.HTML("<br>"),
            widgets.HTML("<h3>Visualization Options</h3>"),
            self.viz_type,
            self.color_by,
            widgets.HTML("<br>"),
            self.update_button,
            self.export_button,
            widgets.HTML("<br>"),
            self.results_text
        ])

        # Display
        display(widgets.HBox([
            filters,
            widgets.HTML("<div style='margin-left: 20px;'></div>")
        ]))

        # Initial visualization
        self._update_visualization()

    def _apply_filters(self) -> List[Dict]:
        """Apply current filters to nodes."""
        filtered = self.visualizer.nodes.copy()

        # Search filter
        if self.search_box.value:
            search_term = self.search_box.value.lower()
            filtered = [
                node for node in filtered
                if search_term in node['narrative'].lower()
            ]

        # Topic filter
        if 'All' not in self.topic_filter.value and self.topic_filter.value:
            selected_topics = set(self.topic_filter.value)
            filtered = [
                node for node in filtered
                if any(topic in selected_topics for topic in node['topics'])
            ]

        # Actor filter
        if 'All' not in self.actor_filter.value and self.actor_filter.value:
            selected_actors = set(self.actor_filter.value)
            filtered = [
                node for node in filtered
                if any(actor in selected_actors for actor in node['actors'])
            ]

        # Article count filter
        min_count = self.article_count_slider.value
        filtered = [
            node for node in filtered
            if node['article_count'] >= min_count
        ]

        return filtered

    def _on_update_click(self, button):
        """Handle update button click."""
        self._update_visualization()

    def _update_visualization(self):
        """Update visualization based on current filters."""
        # Apply filters
        self.filtered_nodes = self._apply_filters()

        # Update results text
        self.results_text.value = (
            f"<b>Showing {len(self.filtered_nodes)} narratives</b> "
            f"(of {len(self.visualizer.nodes)} total)"
        )

        if not self.filtered_nodes:
            print("No narratives match the current filters.")
            return

        # Create temporary filtered visualizer
        filtered_graph_data = {
            'nodes': self.filtered_nodes,
            'macro_arguments': self.visualizer.macro_arguments,
            'statistics': self.visualizer.graph_data.get('statistics', {})
        }

        from .graph_visualizer import NarrativeGraphVisualizer
        temp_viz = NarrativeGraphVisualizer(
            graph_data=filtered_graph_data,
            embedding_model=self.visualizer.embedding_model
        )

        # Generate visualization
        try:
            if self.viz_type.value == '2D Network':
                fig = temp_viz.create_2d_visualization(
                    color_by=self.color_by.value,
                    min_article_count=1  # Already filtered
                )
            elif self.viz_type.value == '3D Network':
                fig = temp_viz.create_3d_visualization(
                    color_by=self.color_by.value,
                    min_article_count=1
                )
            elif self.viz_type.value == 't-SNE Clusters':
                if self.visualizer.embedding_model:
                    fig = temp_viz.create_cluster_view()
                else:
                    print("Embeddings required for t-SNE view. Use 2D/3D Network instead.")
                    return
            elif self.viz_type.value == 'Topic Distribution':
                fig = temp_viz.create_topic_distribution()

            fig.show()

        except Exception as e:
            print(f"Error creating visualization: {e}")

    def _on_export_click(self, button):
        """Handle export button click."""
        import json
        from datetime import datetime

        # Create export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'filters_applied': {
                'search': self.search_box.value,
                'topics': list(self.topic_filter.value),
                'actors': list(self.actor_filter.value),
                'min_article_count': self.article_count_slider.value
            },
            'narrative_count': len(self.filtered_nodes),
            'narratives': self.filtered_nodes
        }

        # Save to file
        filename = f'filtered_narratives_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported {len(self.filtered_nodes)} narratives to {filename}")

    def query_interface(self):
        """Display a simple query interface."""
        # Query widgets
        query_type = widgets.Dropdown(
            options=['By Actor', 'By Topic', 'By Place', 'Composite'],
            description='Query Type:',
            style={'description_width': '120px'}
        )

        query_input = widgets.Text(
            placeholder='Enter query value...',
            description='Value:',
            style={'description_width': '120px'}
        )

        query_button = widgets.Button(
            description='Run Query',
            button_style='info',
            icon='search'
        )

        results_area = widgets.Output()

        def on_query_click(button):
            with results_area:
                results_area.clear_output()

                query_value = query_input.value
                if not query_value:
                    print("Please enter a query value")
                    return

                # Run query
                if query_type.value == 'By Actor':
                    results = self.graph.query_by_actor(query_value)
                elif query_type.value == 'By Topic':
                    results = self.graph.query_by_topic(query_value)
                elif query_type.value == 'By Place':
                    results = self.graph.query_by_place(query_value)
                else:
                    print("Composite queries not yet implemented in this interface")
                    return

                # Display results
                print(f"\n{'=' * 60}")
                print(f"Query Results: {len(results)} narratives found")
                print('=' * 60)

                for i, result in enumerate(results[:20], 1):
                    print(f"\n{i}. {result['narrative']}")
                    print(f"   Articles: {result['article_count']}")
                    print(f"   Article IDs: {result['article_ids'][:10]}")

                if len(results) > 20:
                    print(f"\n... and {len(results) - 20} more results")

        query_button.on_click(on_query_click)

        # Display
        display(widgets.VBox([
            widgets.HTML("<h3>Query Interface</h3>"),
            query_type,
            query_input,
            query_button,
            results_area
        ]))


def create_narrative_dashboard(visualizer, graph):
    """
    Create a comprehensive dashboard with multiple views.

    Args:
        visualizer: NarrativeGraphVisualizer instance
        graph: NarrativeGraph instance
    """
    from IPython.display import display, HTML
    import ipywidgets as widgets

    # Header
    display(HTML("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>📊 Narrative Graph Explorer</h1>
        <p style='color: white; margin: 5px 0 0 0;'>Interactive visualization and analysis</p>
    </div>
    """))

    # Statistics
    stats = graph.get_summary()

    display(HTML(f"""
    <div style='display: flex; gap: 20px; margin-bottom: 20px;'>
        <div style='flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px;'>
            <h3 style='margin: 0 0 10px 0; color: #667eea;'>📈 Total Narratives</h3>
            <p style='font-size: 32px; margin: 0; font-weight: bold;'>{stats['total_narratives']}</p>
        </div>
        <div style='flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px;'>
            <h3 style='margin: 0 0 10px 0; color: #764ba2;'>🎯 Macro-Arguments</h3>
            <p style='font-size: 32px; margin: 0; font-weight: bold;'>{stats['total_macro_arguments']}</p>
        </div>
        <div style='flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px;'>
            <h3 style='margin: 0 0 10px 0; color: #f093fb;'>📰 Avg Articles/Narrative</h3>
            <p style='font-size: 32px; margin: 0; font-weight: bold;'>{stats['avg_articles_per_narrative']:.1f}</p>
        </div>
    </div>
    """))

    # Tabs for different views
    tab_contents = [
        'Interactive Explorer',
        'Quick Visualizations',
        'Query Interface',
        'Top Entities'
    ]

    tab = widgets.Tab()
    tab.children = [
        widgets.Output() for _ in tab_contents
    ]

    for i, title in enumerate(tab_contents):
        tab.set_title(i, title)

    # Tab 1: Interactive Explorer
    with tab.children[0]:
        explorer = InteractiveNarrativeExplorer(visualizer, graph)
        explorer.display()

    # Tab 2: Quick Visualizations
    with tab.children[1]:
        print("Select a visualization:")

        viz_buttons = widgets.VBox([
            widgets.Button(description='2D Network', button_style='info', icon='project-diagram'),
            widgets.Button(description='3D Network', button_style='info', icon='cube'),
            widgets.Button(description='t-SNE Clusters', button_style='info', icon='chart-scatter'),
            widgets.Button(description='Macro-Arguments', button_style='info', icon='th-large'),
            widgets.Button(description='Topic Distribution', button_style='info', icon='chart-bar'),
            widgets.Button(description='Actor Network', button_style='info', icon='users')
        ])

        output_area = widgets.Output()

        def show_viz(button):
            with output_area:
                output_area.clear_output()

                try:
                    if button.description == '2D Network':
                        fig = visualizer.create_2d_visualization()
                    elif button.description == '3D Network':
                        fig = visualizer.create_3d_visualization()
                    elif button.description == 't-SNE Clusters':
                        if visualizer.embedding_model:
                            fig = visualizer.create_cluster_view()
                        else:
                            print("Embedding model required for t-SNE view")
                            return
                    elif button.description == 'Macro-Arguments':
                        fig = visualizer.create_macro_argument_view()
                    elif button.description == 'Topic Distribution':
                        fig = visualizer.create_topic_distribution()
                    elif button.description == 'Actor Network':
                        fig = visualizer.create_actor_network()

                    fig.show()

                except Exception as e:
                    print(f"Error: {e}")

        for button in viz_buttons.children:
            button.on_click(show_viz)

        display(widgets.HBox([viz_buttons, output_area]))

    # Tab 3: Query Interface
    with tab.children[2]:
        explorer = InteractiveNarrativeExplorer(visualizer, graph)
        explorer.query_interface()

    # Tab 4: Top Entities
    with tab.children[3]:
        print("Top Actors:")
        for i, (actor, count) in enumerate(stats['top_actors'][:10], 1):
            print(f"  {i}. {actor}: {count} narratives")

        print("\nTop Topics:")
        for i, (topic, count) in enumerate(stats['top_topics'][:10], 1):
            print(f"  {i}. {topic}: {count} narratives")

        print("\nTop Places:")
        for i, (place, count) in enumerate(stats['top_places'][:10], 1):
            print(f"  {i}. {place}: {count} narratives")

    display(tab)