#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:49:44 2020

@author: barni13
"""


import pandas as pd
import igraph as ig
import numpy as np
import louvain
import itertools as itl
from copy import deepcopy
import matplotlib.pyplot as plt

#functions
def is_integer(n):
    """checks if variable is an integer"""
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()
    
def find_best_new_edge(graph = g, n = 15, k = 3):
    """finds among all possible pairs, n edges with maxiumum modularity.
       recalculating modularities after adding k best edges"""
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
                best_modularities.append(max(modularities))
                pairs.remove(pairs[modularities.index(max(modularities))])
                modularities.remove(max(modularities))
                i += 1
        else:
            for _ in range(n - i):
                graph.add_edge(pairs[modularities.index(max(modularities))][0], pairs[modularities.index(max(modularities))][1])
                best_pairs.append(pairs[modularities.index(max(modularities))])
                best_modularities.append(max(modularities))
                pairs.remove(pairs[modularities.index(max(modularities))])
                modularities.remove(max(modularities))
                i += i      
    return (best_pairs, best_modularities, i)

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
                    if rand < 0.5:
                        influenced_in_k.append(ngbr)
                        all_influenced.append(ngbr)
                        graph.vs.select(name = ngbr)['influenced'] = 1
        arch.append((influenced_in_k, k))                
    return arch, all_influenced    
    


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
g = ig.Graph()


for emp in reportsto['ID']:
    g.add_vertex(name = str(emp))

list_both_sides = []
for k in range(0, len(reportsto['ID'])):
    for l in range(k + 1, len(reportsto['ID'])):
        if mail_counter[k][l] == 0 or mail_counter[l][k] == 0:
            pass
        else:
            both_sides = mail_counter[k][l] + mail_counter[l][k]
            list_both_sides.append(both_sides)
            if both_sides >= 30:
                g.add_edge(str(k), str(l))
            
##########################################################################################        
#                                ustalanie thresholdu                                    #
##########################################################################################
"""
thresholds = []
above_trsh = []        
for threshold in np.linspace(min(list_both_sides), max(list_both_sides), 1000):
    thresholds.append(threshold)
    above_trsh.append(len([x for x in list_both_sides if x >= threshold]))

plt.figure()
plt.plot(thresholds, [x / len(g.vs['name']) for x in above_trsh])

[x / len(g.vs['name']) for x in above_trsh].index(2)

plt.figure()
plt.plot(thresholds[0:18], [x / len(g.vs['name']) for x in above_trsh][0:18])
"""
#########################################################################################


g.delete_vertices(ndls_ids) #verices of former emp/tech. acc. deletion

#difinig of plots visual style
color_dict = ['red', 'blue']
layout = g.layout("fr") #tree / grid_fr / fr / kk / rt_circular
visual_style = {}
visual_style["vertex_size"] = 50
#visual_style["vertex_color"] = [color_dict[int(x == 0)] for x in g.vs.degree()]
visual_style["vertex_label"] = g.vs["name"]
#["edge_length"] = 200
visual_style["edge_width"] = 1
visual_style["layout"] = layout
visual_style["bbox"] = (3000, 2000)
visual_style["margin"] = 50

disconnected = [x['name'] for x in g.vs if x.degree() == 0]
g2 = deepcopy(g)
g2.delete_vertices(disconnected)

ig.plot(g2, **visual_style).save("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/initial_graph.png")

#partition of base graph
partition = louvain.find_partition(g2, louvain.ModularityVertexPartition)
print(partition.quality())
ig.plot(partition, **visual_style).save("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/partition_graph.png")



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

find_best_new_edge(g, 20, 3) 

