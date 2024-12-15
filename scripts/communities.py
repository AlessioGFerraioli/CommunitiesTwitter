# Necessary imports
import pandas as pd
import networkx as nx

# Configurations dictionary for global settings
CONFIG = {
    "datafolder_path": r"D:\\Users\\aless\\Desktop\\Universita\\Complex_Networks_Remondini\\Twitter_proj", # path to read the data

    "datafile": r"D:\\Users\\aless\\Desktop\\Universita\\Complex_Networks_Remondini\\Twitter_proj\\df_annotated_2021-01-26.pkl",
    "processed_file": r"D:\\Users\\aless\\Desktop\\Universita\\Complex_Networks_Remondini\\Twitter_proj\\df_processed_wc2.csv",
    "weight_cut": 2,
    "remove_non_annotated": True,
    "save_processed": False,
    "read_processed": True,
    "output_graph_path": r"D:\\Users\\aless\\Desktop\\output_graph.png"
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

# Step 4: Main Execution
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
    
    # Placeholder for subsequent analysis and visualization steps.
    print("Preprocessing and graph creation complete. Ready for analysis and visualization.")

main()  # Execute the refactored main function.
