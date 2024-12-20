# Necessary imports
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
import leidenalg as la
from datetime import datetime
from collections import defaultdict


# Configurations dictionary for global settings
CONFIG = {
    "datafolder_path": r"D:\\Users\\aless\\Desktop\\Universita\\Complex_Networks_Remondini\\Twitter_proj", # path to read the data

    "datafile": r"D:\\Users\\aless\\Desktop\\Universita\\Complex_Networks_Remondini\\Twitter_proj\\df_annotated_2021-01-26.pkl",
    "processed_file": r"D:\\Users\\aless\\Desktop\\Universita\\Complex_Networks_Remondini\\Twitter_proj\\df_processed_wc2.csv",
    "weight_cut": 2,
    "remove_non_annotated": True,
    "save_processed": False,
    "read_processed": True,
    "output_graph_path": r"D:\\Users\\aless\\Desktop\\output_graph.png",
    "use_louvain": False,  # Set to True for Louvain community detection
    "directed": True,
    "weighted": True
}

# Step 1: Preprocessing
def preprocess_data(config):
    """
    preprocessing the dataframe: 
     - remove tweets for which we don't have a user_annotation
     - remove tweets that are not retweets
     - count how many times each user A retweeted each user B 
     - remove all the weights under 2
     - save the processed dataframe to file
    Args:
        config (dict): Configuration dictionary for paths and flags.
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    print("Starting data preprocessing...")
    # read the datafile and make a dataframe
    df = pd.read_pickle(config["datafile"])
    print(f"Loaded dataset with {len(df)} rows.")

    # Remove rows without annotation if flag is set
    # i remove from my dataframe the rows for which user_annotation is nan
    # (we only want to keep the user with a user_annotation)
    if config["remove_non_annotated"]:
        df = df[~df["user_annotation"].isna()]
        print(f"Rows after removing non-annotated users: {len(df)}.")

    # i remove from my dataframe the rows for which "retweeted_status.id" is nan
    # (we only want to keep the tweets that are retweets)
    df = df.loc[~df["retweeted_status.id"].isna()]
    df["weight"] = df.groupby(["user.id", "retweeted_status.user.id"])["user.id"].transform("size")
    df = df[df["weight"] >= config["weight_cut"]]
    print(f"Rows after applying weight threshold ({config['weight_cut']}): {len(df)}.")

    # Save processed DataFrame if flag is set
    if config["save_processed"]:
        df.to_csv(config["processed_file"])
        print(f"Processed data saved to {config['processed_file']}.")

    return df

# Step 2: Graph Creation
def create_graph(df):
    """
    Creates a directed graph from a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with 'user.id' and 'retweeted_status.user.id' columns.
    Returns:
        nx.DiGraph: A directed graph with weighted edges.
    """
    print("Creating graph from data...")
    G = nx.from_pandas_edgelist(
        df, 
        source="user.id", 
        target="retweeted_status.user.id", 
        edge_attr="weight", 
        create_using=nx.DiGraph()
    )
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# Step 3: Group data into categories
def group_data(df):
    """
    Group data into categories based on user annotations.

    Args:
    df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
    dict: Dictionary containing grouped DataFrames.
    """
    print("Grouping data by user annotations...")
    groups = {
        "AntiVax": df[df["user_annotation"] == "AntiVax"],
        "Neutral": df[df["user_annotation"] == "Neutral"],
        "ProVax": df[df["user_annotation"] == "ProVax"],
    }
    for key, group in groups.items():
        print(f"{key}: {len(group)} rows.")
    return groups


# Step 4: Community Detection
def detect_communities(graph, use_louvain=False, directed=True, weighted=True):
    """
    Detect communities using Leiden or Louvain algorithm.

    Args:
        graph (networkx.Graph): Input graph.
        use_louvain (bool): If True, use Louvain algorithm. If False, use Leiden.
        directed (bool): If True, consider the graph as directed.
        weighted (bool): If True, consider edge weights.

    Returns:
        dict: Dictionary mapping node to community ID.
    """
    print("Converting NetworkX graph to iGraph format...")
    ig_graph = ig.Graph.from_networkx(graph)

    if not directed:
        ig_graph = ig_graph.as_undirected()

    weights = ig_graph.es["weight"] if weighted and "weight" in ig_graph.es.attributes() else None

    print(f"Running {'Louvain' if use_louvain else 'Leiden'} algorithm...")
    if use_louvain:
        partition = ig_graph.community_multilevel(weights=weights)
    else:
        partition = la.find_partition(ig_graph, la.RBConfigurationVertexPartition, weights=weights)

    # Map nodes to community IDs
    node_to_community = {node.index: community for community, nodes in enumerate(partition) for node in nodes}
    print("Community detection completed.")
    return node_to_community


# Step 5: Group Analysis
def analyze_communities(graph, node_to_community):
    """
    Analyze the detected communities for size and centrality.

    Args:
        graph (networkx.Graph): Input graph.
        node_to_community (dict): Mapping of nodes to community IDs.

    Returns:
        dict: Analysis results for each community.
    """
    print("Analyzing communities...")
    community_analysis = defaultdict(lambda: {"size": 0, "nodes": []})

    for node, community in node_to_community.items():
        community_analysis[community]["size"] += 1
        community_analysis[community]["nodes"].append(node)

    for community, info in community_analysis.items():
        subgraph = graph.subgraph(info["nodes"])
        info["degree_centrality"] = nx.degree_centrality(subgraph)

    print("Community analysis completed.")
    return community_analysis


# Step 6: Visualization
def visualize_communities(graph, node_to_community, output_file=None):
    """
    Visualize the graph with nodes colored by community.

    Args:
        graph (networkx.Graph): Input graph.
        node_to_community (dict): Mapping of nodes to community IDs.
        output_file (str): Path to save the visualization. If None, display interactively.
    """
    print("Visualizing communities...")
    pos = nx.spring_layout(graph)
    communities = set(node_to_community.values())

    plt.figure(figsize=(10, 8))
    for community in communities:
        nodes = [node for node, comm in node_to_community.items() if comm == community]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, label=f"Community {community}")

    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    plt.legend()

    if output_file:
        plt.savefig(output_file)
        print(f"Community visualization saved to {output_file}.")
    else:
        plt.show()

def main():
    # load the data (if necessary, pre-process it)
    if CONFIG["read_processed"]:
        # this block is when we already have a saved processed file
        df = pd.read_csv(CONFIG["processed_file"])
        print("Processed data loaded.")
    else:
        # this block is when we don't already have a saved processed file, so we need to do the processing
        df = preprocess_data(CONFIG)
    
    # i create the networkx graph from the dataframe
    G = create_graph(df)
    
    print("Preprocessing and graph creation complete.")

    
    # Detect communities
    print("Starting Community detection..")
    node_to_community = detect_communities(G, use_louvain=CONFIG["use_louvain"], directed=CONFIG["directed"], weighted=CONFIG["weighted"])
    print("Community detection completed.")

    # Analyze communities
    community_analysis = analyze_communities(G, node_to_community)
    print(community_analysis)

    # Visualize communities
    filename = "communities.png"
    visualize_communities(G, node_to_community, output_file=filename)
    print(f"Community visualization saved to {filename}.") 


main()  # Execute the refactored main function.




# Integration Example
def main_with_community_analysis():
    # Load and preprocess the data (placeholder call)
    G = nx.DiGraph()

    # Example: Add edges (this should be replaced with actual data loading)
    G.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 0.8), (3, 4, 1.2), (4, 1, 0.4)])


if __name__ == "__main__":
    main_with_community_analysis()
