import argparse
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_dag(dag_path):
    # Determine output path
    dag_basename = os.path.splitext(os.path.basename(dag_path))[0]
    output_dir = "outputs/adjustment_sets"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dag_basename}.pdf")

    # Load edges from json
    with open(dag_path, 'r') as f:
        edges_list = json.load(f)
    
    # Create directed graph
    causal_graph = nx.DiGraph()
    causal_graph.add_edges_from(edges_list)

    # User provided plot snippet
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    labels={
        'accessibilityRAI': 'Rural\naccess', 'fertilizerTiming': 'Fert.\nTiming', 
        'fertilizerAmountN': 'N-Rate', 'fertilizerAmountP': 'P-Rate',
        'fieldHistory': 'Field\nhistory', 'outcome': 'Crop\nprod.', 
        'soil': 'Soil.','investmentCap': 'Invest.\nCapacity', 'variety': 'Variety',
        'weatherSeason': 'Seasonal\nWeather', 'otherManagement': "Other\nMnmgt",
    }

    edge_colors = []
    for cause, effect in causal_graph.edges:
        if cause in ['soil', 'weather', 'elevation']:
            edge_colors.append("#0037C3")
        elif cause in ('fertilizerAmountN', 'fertilizerAmountP', 'fertilizerTiming', 'manure', 'variety'):
            edge_colors.append('#15a123')
        elif cause in ('previousAdvice', 'neighborAdvice'):
            edge_colors.append('#f58802')
        elif cause in ('investmentCap', 'fieldHistory'):
            edge_colors.append('#fa2616')
        elif cause in ('accessibilityRAI', 'elevation'):
            edge_colors.append("#de04cf")
        elif cause == 'history':
            edge_colors.append('#460080')
        else:
            edge_colors.append('k')

    pos={
        'accessibilityRAI': (-3.8, -0.64), 'cropgrowth': (2.1, 1.15),
        'fertilizerAmountN': (-1.5, .0), 'fertilizerAmountP': (0.5, 2.), 
        'fertilizerTiming': (-1.5, -1.6), 'fieldHistory': (-2.8, 3.5),
        'outcome': (4.1, 1.15), 'soil': (0.5, -1.5), 'otherManagement': (1.8, 3.8),
        'investmentCap': (-3.99, 2.06), 'variety': (-.5, 4.),
        'weatherSeason': (0.5, -2.6)
    }

    # Only pass labels for nodes that are actually in the graph to avoid errors
    node_labels = {node: labels.get(node, node) for node in causal_graph.nodes}
    
    # We also only pass pos for nodes that are in the graph. Any node missing from pos will raise an error.
    # So we'll assign a default position or ensure all graph nodes exist in pos.
    # To be safe against a graph having an unexpected node, we filter:
    plot_pos = {node: pos.get(node, (0, 0)) for node in causal_graph.nodes}

    nx.draw_networkx(
        causal_graph, ax=ax, node_size=2200, 
        labels=node_labels, 
        pos=plot_pos,
        node_shape='o', node_color='#ffffff00', 
        edge_color=edge_colors,
    )
    
    color_labels = [
        ('#0060c7', 'Weather'), ('#15a123', 'Management'), 
        ('#f58802', 'Advice'), ('#fa2616', 'Farmer context'), 
        ('#460080', 'Field history'), ('k', 'Cropping system'), 
        ('#de04cf', 'Farm accessibility')
    ]
    ax.legend(handles=[
        mpl.lines.Line2D([], [], color=color, linestyle='-', linewidth=2, label=label)
        for color, label in color_labels
    ], loc='lower right', title='Effects legend')
    
    ax.set_title('Causal Diagram', fontweight='bold', fontsize=15)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
        
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved DAG plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a JSON DAG definition.")
    parser.add_argument('dag_path', type=str, help="Path to the JSON DAG file (e.g. config/dags/main-rev.json)")
    args = parser.parse_args()
    
    plot_dag(args.dag_path)
