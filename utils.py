# filename : contains input filename with full path 
# file should have at least two columns named 'subject' and 'object'
# rootnode: specify the url for root node of the hierarchy
# target_filename: save embeddings with key name
# lambda_factor : tuneable factor to create embeddings, read http://ceur-ws.org/Vol-2600/paper16.pdf


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timeit
import itertools 
import operator
import csv
from sklearn import tree, linear_model
import matplotlib.pyplot as plt
import math
import itertools
from multiprocessing import Pool, cpu_count
import os
import networkx as nx
import glob
from nltk.corpus import wordnet as wn
import logging

# extract graph from wordnet
def closure_graph_bottomup(synset_list, fn):
    seen = set()
    graph = nx.DiGraph()
    for synset in synset_list:
        def recurse(s):
            if not s in seen:
                seen.add(s)
                graph.add_node(s.name())
                for s1 in fn(s):
                    graph.add_node(s1.name())
                    graph.add_edge(s1.name(), s.name())
                    recurse(s1)

        recurse(synset)
    return graph
    

    
# create embeddings based on wordnet graph : one word may have multiple parent concepts
# Input parameters
# filename : csv file containg graph edges : columns parent, child
# target_filename : specify the csv file name and location to save embeddings
# lambda_factor : used in creating embeddings
# concept_dict : dictionary containing words as keys and list of synsets for the given words
def create_embeddings_wordnet(filename, rootnode,target_filename, lambda_factor=0.7, concept_dict=None):
    df = pd.read_csv(filename)
    # Create the Directed Graph 
    try:
        G = nx.from_pandas_edgelist(df,
                            source='parent',
                            target='child',
                            create_using=nx.DiGraph())
        # find level of node(shortest path from root to current node)
        optional_attrs = nx.shortest_path_length(G ,rootnode)
        nx.set_node_attributes(G ,  optional_attrs, 'node_level' )
        
        word_pairs = list(itertools.product(list(concept_dict.keys()), repeat=2)) # create pair of all nodes 

         # get lowest common ancestors of alll pairs of nodes
        
        ancestors_distance = {}
        for word in word_pairs:
            common_ancestor = {} 
            # fetch synsets from the graph for both words
            concepts_word1 = concept_dict[word[0]]
            concepts_word2 = concept_dict[word[1]]
            pairs = list(itertools.product(concepts_word1, concepts_word2)) # create pair of all nodes 
            
            ls_similarity = []
            
            for i in pairs:
                
                preds_1 = list (nx.ancestors(G, i[0]))
                preds_2 = list( nx.ancestors(G, i[1]))
                common_preds = list( set([n for n in preds_1]).intersection(set([n for n in preds_2])))
                p_dict = {}
                [ p_dict.update( {parent : G.nodes[parent]['node_level']}) for parent in common_preds ]
                common_ancestor[i] = max(p_dict.items(), key=operator.itemgetter(1))[0]
                
            # replace ancestor node with max distance from ancestor to the nodes
            distance_list = []
            
            for key in common_ancestor:
                 distance_list.append(np.max( [ nx.shortest_path_length(G,common_ancestor[key],key[0]) ,
                                nx.shortest_path_length(G, common_ancestor[key],key[1]) ] ) )
            ancestors_distance[word] = max( distance_list)
            


        chunked_data = [[k[0],k[1], v] for k, v in ancestors_distance.items()]
        df_nodes = pd.DataFrame(chunked_data)
        df_nodes = df_nodes.rename(columns= {0:'node1', 1:'node2', 2:'weight'})
        depth = df_nodes.weight.max()-1 # find the maximum levels in the hierarchy

        # create adjancey matrix
        vals = np.unique(df_nodes[['node1', 'node2']])
        df_nodes = df_nodes.pivot(index='node1', columns='node2', values='weight'
                          ).reindex(columns=vals, index=vals, fill_value=0)

        df_adjacency = df_nodes.apply( lambda x:   lambda_factor** x)

        # set diagnoal to 1
        pd.DataFrame.set_diag = set_diag
        df_adjacency.set_diag(1)
        df_adjacency.fillna(0, inplace=True)


        df_adjacency.to_csv(target_filename)

    except BaseException:
        logging.exception("An exception was thrown!")

            
    
    
# create embeddings based on  tree/single hierarchy for unbalanced tree
def create_embeddings_unbalanced(filename, rootnode, target_filename, lambda_factor=0.7,  ls_leafnodes=None):
    df = pd.read_csv(filename)
    # Create the Directed Graph 
    try:
        G = nx.from_pandas_edgelist(df,
                            source='parent',
                            target='child',
                            create_using=nx.DiGraph())
    except KeyError:
        G = nx.from_pandas_edgelist(df,
                            source='object',
                            target='subject',
                            create_using=nx.DiGraph())
    # create tree by specifying root node
    tree = nx.bfs_tree(G, rootnode) #
    # find level of node(shortest path from root to current node)
    optional_attrs = nx.shortest_path_length(tree ,rootnode)
    nx.set_node_attributes(tree ,  optional_attrs, 'node_level' )
    
    # to retrieve all nodes in the hierachy
    #ls_leafnodes = [node for node in tree.nodes()]
    
    # to retrieve only leaf nodes
    #ls_leafnodes = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]
    
    #original data may occur from various levels in the hierarchy
    if ls_leafnodes==[]:
        ls_leafnodes = df['child'].unique()
    pairs = list(itertools.product(ls_leafnodes, repeat=2)) # create pair of all nodes 
    all_ancestors = nx.algorithms.all_pairs_lowest_common_ancestor(tree, pairs=pairs) # get lowest common ancestors of alll pairs of nodes


    # replace ancestor node with max distance from ancestor to the nodes
    # 
    ls_ancestors_distance = {}
    for i in all_ancestors:
        ls_ancestors_distance[i[0]] = np.max( [ nx.shortest_path_length(tree,i[1],i[0][0]) ,
                     nx.shortest_path_length(tree, i[1],i[0][1]) ] ) 
        
    chunked_data = [[k[0],k[1], v] for k, v in ls_ancestors_distance.items()]
    df_nodes = pd.DataFrame(chunked_data)
    df_nodes = df_nodes.rename(columns= {0:'node1', 1:'node2', 2:'weight'})
    depth = df_nodes.weight.max() # find the maximum levels in the hierarchy

    # create adjancey matrix
    vals = np.unique(df_nodes[['node1', 'node2']])
    df_nodes = df_nodes.pivot(index='node1', columns='node2', values='weight'
                      ).reindex(columns=vals, index=vals, fill_value=0)

    df_adjacency = df_nodes.apply( lambda x:  lambda_factor**  x)
    #df_adjacency = df_nodes.apply( lambda x:  np.exp( - (depth - x)/5))


    # set diagnoal to 1
    pd.DataFrame.set_diag = set_diag
    df_adjacency.set_diag(1)
    df_adjacency.fillna(0, inplace=True)

    df_adjacency.to_csv(target_filename)

    
    
    
    
    
    
# create embeddings based on unbalanced poly-hierarchies/forest
def create_embeddings_unbalanced_forest(filename, rootnode,target_filename, lambda_factor=0.7, ls_leafnodes=None):
    df = pd.read_csv(filename)
    # Create the Directed Graph 
    G = nx.from_pandas_edgelist(df,
                            source='parent',
                            target='child',
                            create_using=nx.DiGraph())
    
    # find level of node(shortest path from root to current node)
    optional_attrs = nx.shortest_path_length(G ,rootnode)
    nx.set_node_attributes(G ,  optional_attrs, 'node_level' )
    
    # to retrieve all nodes in the forest
    #ls_leafnodes = [node for node in G.nodes()]
    
    # to retrieve only leaf nodes  in the forest
    #ls_leafnodes = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
    if ls_leafnodes==[]:
        ls_leafnodes = df['child'].unique()
    pairs = list(itertools.product(ls_leafnodes, repeat=2)) # create pair of all nodes 
    
     # get lowest common ancestors of alll pairs of nodes
    ls_ancestors_forest = {}
    for i in pairs:
        preds_1 = list (nx.ancestors(G, i[0]))
        preds_2 = list( nx.ancestors(G, i[1]))
        common_preds = list( set([n for n in preds_1]).intersection(set([n for n in preds_2])))
        p_dict = {}
        [ p_dict.update( {parent : G.nodes[parent]['node_level']}) for parent in common_preds ]

        ls_ancestors_forest[i] = max(p_dict.items(), key=operator.itemgetter(1))[0]
        
    
    # replace ancestor node with max distance from ancestor to the nodes
    # 
    ls_ancestors_distance = {}
    for key in ls_ancestors_forest:
        ls_ancestors_distance[key] = np.max( [ nx.shortest_path_length(G,ls_ancestors_forest[key],key[0]) ,
                    nx.shortest_path_length(G, ls_ancestors_forest[key],key[1]) ] ) 


        
    chunked_data = [[k[0],k[1], v] for k, v in ls_ancestors_distance.items()]
    df_nodes = pd.DataFrame(chunked_data)
    df_nodes = df_nodes.rename(columns= {0:'node1', 1:'node2', 2:'weight'})
    depth = df_nodes.weight.max()-1 # find the maximum levels in the hierarchy
    
    # create adjancey matrix
    vals = np.unique(df_nodes[['node1', 'node2']])
    df_nodes = df_nodes.pivot(index='node1', columns='node2', values='weight'
                      ).reindex(columns=vals, index=vals, fill_value=0)

    df_adjacency = df_nodes.apply( lambda x:   lambda_factor** x)

    # set diagnoal to 1
    pd.DataFrame.set_diag = set_diag
    df_adjacency.set_diag(1)
    df_adjacency.fillna(0, inplace=True)

    
    df_adjacency.to_csv(target_filename)



    
def set_diag(self, values): 
    n = min(len(self.index), len(self.columns))
    self.values[tuple([np.arange(n)] * 2)] = values

