#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:32:31 2020

@author: barni13
"""
from datetime import datetime
import pandas as pd
import igraph as ig
import numpy as np
import louvain
import itertools as itl
from copy import deepcopy

#functions
def is_integer(n):
    """checks if variable is an integer"""
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()
    
def find_best_new_edges(graph, n = 15, k = 3):
    """finds among graph' all pairs without connection, n edges with maxiumum modularity.
       recalculating modularities after adding k best edges.
       
       Function call adds edges to graph given as a parametr"""
    pairs = list(itl.combinations(graph.vs['name'], 2))
    edges_in_graph = [(x['name'], y['name']) for (x, y) in [e.vertex_tuple for e in graph.es]]
    for pair in edges_in_graph:
        pairs.remove(pair)
    
    best_pairs = []
    best_modularities = []
    
    i = 0
    while i < n:
        modularities = []
        for m in range(len(pairs)):
            new_graph = deepcopy(graph)
            new_graph.add_edge(pairs[m][0], pairs[m][1])
            modularities.append(louvain.find_partition(new_graph, louvain.ModularityVertexPartition).quality())
        if n - i > k:
            for _ in range(k):
                graph.add_edge(pairs[modularities.index(max(modularities))][0], pairs[modularities.index(max(modularities))][1])
                best_pairs.append(pairs[modularities.index(max(modularities))])
                best_modularities.append(louvain.find_partition(graph, louvain.ModularityVertexPartition).quality())
                pairs.remove(pairs[modularities.index(max(modularities))])
                modularities.remove(max(modularities))
                i += 1
        else:
            for _ in range(n - i):
                graph.add_edge(pairs[modularities.index(max(modularities))][0], pairs[modularities.index(max(modularities))][1])
                best_pairs.append(pairs[modularities.index(max(modularities))])
                best_modularities.append(louvain.find_partition(graph, louvain.ModularityVertexPartition).quality())
                pairs.remove(pairs[modularities.index(max(modularities))])
                modularities.remove(max(modularities))
                i += 1
    return (best_pairs, best_modularities, i)

def find_random_new_edges(graph, n = 20):
    """finds among graph' all pairs without connection, n random edges.
       Function call adds edges to graph given as a parametr"""
    random_pairs = []   
       
    pairs = list(itl.combinations(graph.vs['name'], 2))
    edges_in_graph = [(x['name'], y['name']) for (x, y) in [e.vertex_tuple for e in graph.es]]
    for pair in edges_in_graph:
        pairs.remove(pair)
    for rand in np.random.choice(len(pairs) - 1, n, replace = False):
        random_pairs.append(pairs[rand])
        graph.add_edge(pairs[rand][0], pairs[rand][1])
    return random_pairs
        
        
    
        
    

def independent_cascades(graph, initial):
    k = 0
    arch = [(initial, k)]
    all_influenced = initial
    influenced_in_k = initial
    for emp in initial:
        graph.vs.select(name = emp)['influenced'] = 1
    while influenced_in_k != [] and k < 10:
        k = k + 1
        influenced_in_k = []
        for emp in initial:
            for ngbr in [graph.vs.select(x)['name'][0] for x in graph.neighbors(emp, mode = 'out')]:
                if ngbr not in all_influenced:
                    rand = np.random.uniform()
                    if rand < graph.es[graph.get_eid(emp, ngbr)]['weights']:
                        influenced_in_k.append(ngbr)
                        all_influenced.append(ngbr)
                        graph.vs.select(name = ngbr)['influenced'] = 1
        arch.append((influenced_in_k, k))                
    return arch, all_influenced    
    