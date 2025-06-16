import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.metrics import mutual_info_score

class MyPrivBayes:
    """
    A custom implementation of the PrivBayes algorithm for differentially private synthetic data generation.
    
    This class implements the PrivBayes algorithm which:
    1. Learns a Bayesian network structure from the data
    2. Computes differentially private conditional probability distributions
    3. Samples synthetic records from the learned model
    
    Attributes:
        epsilon (float): Total privacy budget for the algorithm
        network (nx.DiGraph): Learned Bayesian network structure
        conditional_probabilities (dict): Private conditional probability distributions
        nodes (list): Attribute order for sampling
        root_node (str): Root node of the network
    """
    
    def __init__(self, epsilon):
        """
        Initialize the MyPrivBayes synthesizer.
        
        Args:
            epsilon (float): Total privacy budget for the algorithm
        """
        self.epsilon = epsilon
        self.network = None
        self.conditional_probabilities = None
        self.nodes = None
        self.root_node = None
    
    def learn_network(self, df):
        """
        Learn the Bayesian network structure using the Chow-Liu algorithm.
        
        This method implements the Chow-Liu algorithm to learn a tree-structured
        Bayesian network by:
        1. Computing mutual information between all pairs of attributes
        2. Creating a fully connected graph with mutual information as edge weights
        3. Finding the maximum spanning tree of this graph
        
        Args:
            df (pd.DataFrame): Input dataset to learn from
        """
        # Get list of attributes (columns)
        attributes = df.columns.tolist()
        
        # Initialize an undirected graph
        G = nx.Graph()
        
        # Add all attributes as nodes
        G.add_nodes_from(attributes)
        
        # Calculate mutual information between all pairs of attributes
        for col1, col2 in combinations(attributes, 2):
            # Calculate mutual information
            mi = mutual_info_score(df[col1], df[col2])
            # Add edge with mutual information as weight
            G.add_edge(col1, col2, weight=mi)
        
        # Find maximum spanning tree
        mst = nx.maximum_spanning_tree(G)
        
        # Choose root node (node with highest degree in the MST)
        degrees = dict(mst.degree())
        self.root_node = max(degrees.items(), key=lambda x: x[1])[0]
        
        # Convert to directed graph by orienting edges away from root
        self.network = nx.DiGraph()
        self.network.add_nodes_from(mst.nodes())
        
        # Use BFS to orient edges away from root
        visited = set()
        queue = [self.root_node]
        visited.add(self.root_node)
        
        while queue:
            current = queue.pop(0)
            for neighbor in mst.neighbors(current):
                if neighbor not in visited:
                    # Orient edge from current to neighbor
                    self.network.add_edge(current, neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Store nodes in topological order (BFS order from root)
        self.nodes = list(nx.bfs_tree(self.network, source=self.root_node))
    
    def compute_private_conditionals(self, df):
        """
        Compute differentially private conditional probability distributions.
        
        This method computes the conditional probability distributions for each attribute
        in the learned network while ensuring differential privacy through Laplace noise.
        
        Args:
            df (pd.DataFrame): Input dataset to learn from
        """
        # Get traversal order using BFS from root
        traversal_order = list(nx.bfs_tree(self.network, source=self.root_node))
        
        # Initialize dictionary to store conditional probabilities
        self.conditional_probabilities = {}
        
        # Allocate privacy budget
        n_attributes = len(self.nodes)
        epsilon_per_conditional = self.epsilon / (2 * (n_attributes - 1)) if n_attributes > 1 else self.epsilon
        epsilon_root = self.epsilon / 2 if n_attributes > 1 else self.epsilon
        
        # Process each node in traversal order
        for node in traversal_order:
            if node == self.root_node:
                # For root node, compute marginal probabilities
                counts = df[node].value_counts()
                # Add Laplace noise
                noisy_counts = counts + np.random.laplace(0, 1/epsilon_root, size=len(counts))
                # Clip negative values to 0
                noisy_counts[noisy_counts < 0] = 0
                # Normalize to get probabilities
                self.conditional_probabilities[node] = noisy_counts / noisy_counts.sum()
            else:
                # For non-root nodes, compute conditional probabilities
                parent = list(self.network.predecessors(node))[0]
                # Calculate conditional counts
                conditional_counts = df.groupby([parent, node]).size().unstack(fill_value=0)
                # Add Laplace noise
                noisy_counts = conditional_counts + np.random.laplace(
                    0, 1/epsilon_per_conditional, size=conditional_counts.shape
                )
                # Clip negative values to 0
                noisy_counts[noisy_counts < 0] = 0
                # Normalize each row to get conditional probabilities
                row_sums = noisy_counts.sum(axis=1)
                # Handle zero rows by setting uniform distribution
                zero_rows = row_sums == 0
                if zero_rows.any():
                    n_categories = noisy_counts.shape[1]
                    noisy_counts.loc[zero_rows] = 1/n_categories
                    row_sums[zero_rows] = 1
                # Normalize
                self.conditional_probabilities[node] = noisy_counts.div(row_sums, axis=0)
    
    def fit(self, df):
        """
        Learn the Bayesian network structure and compute private conditional probabilities.
        
        Args:
            df (pd.DataFrame): Input dataset to learn from
        """
        self.learn_network(df)
        self.compute_private_conditionals(df)
    
    def sample(self, n_records):
        """
        Generate synthetic records using the learned model.
        
        This method generates synthetic records by sampling from the learned
        Bayesian network and its private conditional probability distributions.
        
        Args:
            n_records (int): Number of synthetic records to generate
            
        Returns:
            pd.DataFrame: Generated synthetic dataset with the same column order as the original data
        """
        # Get traversal order using BFS from root
        traversal_order = list(nx.bfs_tree(self.network, source=self.root_node))
        
        # Initialize list to store synthetic records
        synthetic_data = []
        
        # Generate n_records synthetic rows
        for _ in range(n_records):
            # Initialize empty dictionary for current record
            record = {}
            
            # Sample each node in traversal order
            for node in traversal_order:
                if node == self.root_node:
                    # For root node, sample from marginal distribution
                    probs = self.conditional_probabilities[node]
                    sampled_value = np.random.choice(probs.index, p=probs.values)
                    record[node] = sampled_value
                else:
                    # For non-root nodes, sample from conditional distribution
                    parent = list(self.network.predecessors(node))[0]
                    parent_value = record[parent]
                    conditional_probs_df = self.conditional_probabilities[node]
                    
                    # Get probability distribution for current parent value
                    probs = conditional_probs_df.loc[parent_value]
                    
                    # Sample value using conditional probabilities
                    sampled_value = np.random.choice(probs.index, p=probs.values)
                    record[node] = sampled_value
            
            # Add completed record to synthetic data
            synthetic_data.append(record)
        
        # Convert list of dictionaries to DataFrame
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Ensure columns are in the same order as original data
        synthetic_df = synthetic_df[self.nodes]
        
        return synthetic_df 