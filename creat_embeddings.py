import re
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from utils import closure_graph_bottomup, create_embeddings_wordnet
import itertools
import logging
import operator
import numpy as np


word_file = 'bm_allsynsets_dictionary.npy'
file = 'BM_all_synsets.csv'
target_file = 'BM_all_synsetsEmbedding.csv'
root_node = 'root'
lambda_factor = 0.7



def main():
    
    syn_dict =  np.load(word_file,allow_pickle='TRUE').item()
    create_embeddings_wordnet(file,  root_node , target_file,lambda_factor, syn_dict)




if __name__ == '__main__':
    main()


