#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:49:44 2020

@author: barni13
"""

import pandas as pd
import igraph as ig
import numpy as np
import leidenalg
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
    
def find_best_new_edge(graph, n):
    """finds among all possible pairs n edges with maxiumum modularity"""
    pairs = list(itl.permutations(graph.vs['name'], 2))
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
            new_graph.add_edge(pairs[m][0], pairs[m][1], weights = np.mean(graph.es['weights']))
            modularities.append(leidenalg.find_partition(new_graph, leidenalg.ModularityVertexPartition).quality)
        print(modularities)
        graph.add_edge(pairs[modularities.index(max(modularities))][0], pairs[modularities.index(max(modularities))][1], weights = np.mean(graph.es['weights']))
        best_pairs.append(pairs[modularities.index(max(modularities))])
        best_modularities.append(max(modularities))
        pairs.remove(pairs[modularities.index(max(modularities))])
        i = i + 1
    return (best_pairs, best_modularities)
    


#load data
communication = pd.read_csv("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/Dane/communication.csv", sep =";")
reportsto = pd.read_csv("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/Dane/reportsto.csv", sep = ";")
print(communication) 
print(reportsto)

#convert to index with start at 0
communication['Sender'] = communication['Sender'] - 1
communication['Recipient'] = communication['Recipient'] - 1

reportsto['ID'] = reportsto['ID'] - 1
reportsto['ReportsToID'] = [int(x) - 1 if is_integer(x) else x for x in reportsto['ReportsToID']]

print(communication) 
print(reportsto)

#needless email accounts' IDs
ndls_ids = list(reportsto[reportsto['ReportsToID'].isin(['former employee account', 'technical email account - not used by employees'])]['ID'])

 #"reportsto" containing only active employees 
act_emp_rep = reportsto[~reportsto['ID'].isin(ndls_ids)] 

#mails from/to active employees only
act_emp_mails = communication[communication['Sender'].isin(ndls_ids).apply(int) + communication['Recipient'].isin(ndls_ids).apply(int) == 0]

#mails counter
mail_counter = np.zeros((len(reportsto['ID']), len(reportsto['ID'])))

for i in range(0, len(communication)):
    mail_counter[communication.iloc[i]['Sender']][communication.iloc[i]['Recipient']] += 1
    
#mails counter -> matrix with weights
weights = mail_counter/mail_counter.sum(axis=1, keepdims=True)
weights = np.nan_to_num(weights)

#graph, vertices and edges
g = ig.Graph(directed = True)

for emp in reportsto['ID']:
    g.add_vertex(name = str(emp))

for k in range(0, len(reportsto['ID'])):
    for l in range(0, len(reportsto['ID'])):
        if k == l:
            pass
        elif weights[k][l] != 0:
            g.add_edge(str(k), str(l), weights = weights[k][l])

g.delete_vertices(ndls_ids) #verices of former emp/tech. acc. deletion

#difinig of plots visual style
layout = g.layout("tree") #tree / grid_fr      kk (?)
visual_style = {}
visual_style["vertex_size"] = 20
#visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
#visual_style["vertex_label"] = g.vs["name"]
#visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
visual_style["layout"] = layout
visual_style["bbox"] = (1200, 1200)
visual_style["margin"] = 20

#ig.plot(g, **visual_style).show()

#partition of base graph
partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
print(partition.quality)
#ig.plot(partition, **visual_style).show()

"""
#edge appending
pairs = list(itl.permutations(g.vs['name'], 2)) #all possible pairs of 154 active employees
modularities = []

for m in range(len(pairs)):
    g_new = deepcopy(g)
    g_new.add_edge(pairs[m][0], pairs[m][1], weights = np.mean(g.es['weights']))
    modularities.append(leidenalg.find_partition(g_new, leidenalg.ModularityVertexPartition).modularity)

max(modularities)
pairs[modularities.index(max(modularities))]
"""

find_best_new_edge(g, 3)
