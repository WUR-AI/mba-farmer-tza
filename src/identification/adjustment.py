import json
import networkx as nx

# Monkey patch for networkx d_separated, needed by some dowhy versions
nx.algorithms.d_separated = nx.algorithms.d_separation.is_d_separator
if not hasattr(nx, 'd_separated'):
    nx.d_separated = nx.algorithms.d_separation.is_d_separator

from dowhy.causal_identifier import BackdoorAdjustment, AutoIdentifier, EstimandType

def identify_adjustment_set(edges_list: str, treatment_name: str, outcome_name: str, 
                            adjustment_set_type: str = "OPTIMAL_MINIMUM_ADJ",
                            conditional_nodes: list = None):
    """
    Reads a DAG from a JSON file and identifies the required adjustment set.
    
    Args:
        dag_path (str): Path to the JSON file containing the DAG edges (list of [source, target]).
        treatment_name (str): Name of the treatment node.
        outcome_name (str): Name of the outcome node.
        adjustment_set_type (str): Type of adjustment set. 
            Options: 'MINIMAL', 'EFFICIENT', 'OPTIMAL_MINIMUM_ADJ', 'ALT_MINIMUM_ADJ'
        conditional_nodes (list): Nodes to condition on (e.g., heterogeneous effect modifiers).
        
    Returns:
        list: The sorted list of nodes in the adjustment set.
    """
    if conditional_nodes is None:
        conditional_nodes = []
    causal_graph = nx.DiGraph(edges_list)
    observed_nodes = list(causal_graph.nodes())
    
    try:
        bd_adj = getattr(BackdoorAdjustment, adjustment_set_type)
    except AttributeError:
        valid_options = [attr for attr in dir(BackdoorAdjustment) if not attr.startswith('_')]
        raise ValueError(f"'{adjustment_set_type}' is not a valid BackdoorAdjustment. Valid options are: {valid_options}")
        
    identifier = AutoIdentifier(
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
        backdoor_adjustment=bd_adj,
    )
    
    identified_estimand = identifier.identify_effect(
        graph=causal_graph,
        action_nodes=treatment_name,
        outcome_nodes=outcome_name,
        observed_nodes=observed_nodes,
        conditional_node_names=conditional_nodes
    )
    
    if adjustment_set_type == "BACKDOOR_EXHAUSTIVE":
        # Get descendants of the treatment to exclude them from the required conditional nodes
        descendants = nx.descendants(causal_graph, treatment_name)
        filtered_conditional_nodes = [node for node in conditional_nodes if node not in descendants]
        
        all_sets = list(identified_estimand.backdoor_variables.values())
        valid_sets = []
        for s in all_sets:
            if s is not None and all(c in s for c in filtered_conditional_nodes):
                sorted_s = sorted(s)
                if sorted_s not in valid_sets:
                    valid_sets.append(sorted_s)
        return valid_sets
    else:
        return sorted(identified_estimand.get_backdoor_variables())
