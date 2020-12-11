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

for k in range(0, len(reportsto['ID'])):
    for l in range(k + 1, len(reportsto['ID'])):
        if mail_counter[k][l] == 0 or mail_counter[l][k] == 0:
            pass
        else:
            both_sides = mail_counter[k][l] + mail_counter[l][k]
            list_both_sides.append(both_sides)
            if both_sides >= 20:
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
plt.yscale('log')
plt.show()

[x / len(g.vs['name']) for x in above_trsh].index(2)

plt.figure()
plt.plot(thresholds[0:20], [x / len(g.vs['name']) for x in above_trsh][0:20])
plt.yscale('log')
plt.show()

plt.figure()
plt.plot(thresholds[0:400], [x / len(g.vs['name']) for x in above_trsh][0:400])
plt.yscale('log')
plt.show()
"""
#########################################################################################


g.delete_vertices(ndls_ids) #verices of former emp/tech. acc. deletion

#difinig visual style of graphs' plots 
color_dict = ['red', 'blue']
layout = g.layout("fr") #tree / grid_fr / fr / kk / rt_circular
visual_style = {}
visual_style["vertex_size"] = 50
#visual_style["vertex_color"] = [color_dict[int(x == 0)] for x in g.vs.degree()]
visual_style["vertex_label"] = g.vs["name"]
visual_style["edge_width"] = 1
visual_style["layout"] = layout
visual_style["bbox"] = (3000, 2000)
visual_style["margin"] = 50

disconnected = [x['name'] for x in g.vs if x.degree() == 0]
g2 = deepcopy(g)
g2.delete_vertices(disconnected)

ig.plot(g, **visual_style, mark_groups = True).show()


#.save("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/initial_graph.png")

#partition of base graph
partition = louvain.find_partition(g2, louvain.ModularityVertexPartition)
print(partition.quality())
ig.plot(partition, **visual_style, mark_groups = True).show() #.save("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/partition_initial_graph.png")



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

partition = louvain.find_partition(g, louvain.ModularityVertexPartition)
print(partition.quality())

g3 = deepcopy(g)
g4 = deepcopy(g)

best_edges = find_best_new_edges(g3, 100, 10)
random_edges = find_random_new_edges(g4, 100)

#partition of modified graph
partition2 = louvain.find_partition(g3, louvain.ModularityVertexPartition)
print(partition2.quality())
ig.plot(partition2, **visual_style).save("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/partition_modified_graph.png")

#partition of randomly modified graph
partition3 = louvain.find_partition(g4, louvain.ModularityVertexPartition)
print(partition3.quality())
ig.plot(partition3, **visual_style).save("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/partition_randomly_modified_graph.png")




# 5% -> -10, 10% -> -19
di_g = ig.Graph(directed = True)
di_g.add_vertices(list(g.vs["name"]))

for x in g.es:
    di_g.add_edge(x.tuple[0], x.tuple[1], weights = weights[x.tuple[0]][x.tuple[1]])
    di_g.add_edge(x.tuple[1], x.tuple[0], weights = weights[x.tuple[1]][x.tuple[0]])
    
    
di_g.vs["closeness"] = di_g.closeness()
di_g.vs["betweenness"] = di_g.betweenness() 
#di_g.vs["clustering_coef] = ...
di_g.vs["indegree"] = di_g.indegree()
di_g.vs["outdegree"] = di_g.outdegree()
di_g.vs["degree"] = di_g.degree()
    
# 5% of the best nodes
di_g_1 = deepcopy(di_g)
closeness5 = di_g_1.vs.select(lambda x:x["closeness"] > sorted(di_g_1.closeness())[-8])
closeness5_IC = independent_cascades(di_g_1, closeness5["name"])
print(len(closeness5_IC[0]), len(closeness5_IC[1]))

di_g_2 = deepcopy(di_g)
betweenness5 = di_g_2.vs.select(lambda x:x["betweenness"] > sorted(di_g_2.betweenness())[-8])
betweenness5_IC = independent_cascades(di_g_2, betweenness5["name"])
print(len(betweenness5_IC[0]), len(betweenness5_IC[1]))

di_g_3 = deepcopy(di_g)
indegree5 = di_g_3.vs.select(lambda x:x["indegree"] > sorted(di_g_3.indegree())[-8])
indegree5_IC = independent_cascades(di_g_3, indegree5["name"])
print(len(indegree5_IC[0]), len(indegree5_IC[1]))

di_g_4 = deepcopy(di_g)
outdegree5 = di_g_4.vs.select(lambda x:x["outdegree"] > sorted(di_g_4.outdegree())[-8])
outdegree5_IC = independent_cascades(di_g_4, indegree5["name"])
print(len(outdegree5_IC[0]), len(outdegree5_IC[1]))

di_g_5 = deepcopy(di_g)
degree5 = di_g_5.vs.select(lambda x:x["degree"] > sorted(di_g_5.degree())[-8])
degree5_IC = independent_cascades(di_g_5, indegree5["name"])
print(len(degree5_IC[0]), len(degree5_IC[1]))

# 10% of the best nodes

di_g_6 = deepcopy(di_g)
closeness10 = di_g.vs.select(lambda x:x["closeness"] > sorted(di_g.closeness())[-16])
closeness10_IC = independent_cascades(di_g_6, closeness10["name"])
print(len(closeness10_IC[0]), len(closeness10_IC[1]))

di_g_7 = deepcopy(di_g)
betweenness10 = di_g.vs.select(lambda x:x["betweenness"] > sorted(di_g.betweenness())[-16])
betweenness10_IC = independent_cascades(di_g_7, betweenness10["name"])
print(len(betweenness10_IC[0]), len(betweenness10_IC[1]))

di_g_8 = deepcopy(di_g)
indegree10 = di_g.vs.select(lambda x:x["indegree"] > sorted(di_g.indegree())[-16])
indegree10_IC = independent_cascades(di_g_8, indegree10["name"])
print(len(indegree10_IC[0]), len(indegree10_IC[1]))

di_g_9 = deepcopy(di_g)
outdegree10 = di_g.vs.select(lambda x:x["outdegree"] > sorted(di_g.outdegree())[-16])
outdegree10_IC = independent_cascades(di_g_9, indegree10["name"])
print(len(outdegree10_IC[0]), len(outdegree10_IC[1]))

di_g_10 = deepcopy(di_g)
degree10 = di_g.vs.select(lambda x:x["degree"] > sorted(di_g.degree())[-16])
degree10_IC = independent_cascades(di_g_10, indegree10["name"])
print(len(degree10_IC[0]), len(degree10_IC[1]))