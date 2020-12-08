#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:30:31 2020

@author: barni13
"""
from datetime import datetime
import pandas as pd
import igraph as ig
import numpy as np
import louvain
import itertools as itl
from copy import deepcopy
import matplotlib.pyplot as plt
from needed_functions import *              #find_best_new_edges, find_random_new_edges, independent_cascades


def wrapper(threshold = 20, N = 50, groups = 5):
    result = [datetime.now().strftime("%d-%m-%y %H:%M"), ("threshold", threshold), ("Max nowych krawędzi", N), ("Ile krawędzi jest dodawanych w jednej iteracji:", groups)]
    
    #load data
    communication = pd.read_csv("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/Dane/communication.csv", sep =";")
    reportsto = pd.read_csv("/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/Dane/reportsto.csv", sep = ";")
    
    #convert to index with start at 0
    communication['Sender'] = communication['Sender'] - 1
    communication['Recipient'] = communication['Recipient'] - 1
    
    reportsto['ID'] = reportsto['ID'] - 1
    reportsto['ReportsToID'] = [int(x) - 1 if is_integer(x) else x for x in reportsto['ReportsToID']]

    mail_counter = np.zeros((len(reportsto['ID']), len(reportsto['ID'])))

    for i in range(0, len(communication)):
        mail_counter[communication.iloc[i]['Sender']][communication.iloc[i]['Recipient']] += 1
        
    #mails counter -> matrix with weights
    weights = mail_counter/mail_counter.sum(axis=1, keepdims=True)
    weights = np.nan_to_num(weights)
    
    
    #main graph, vertices and edges
    g = ig.Graph()
    
    for emp in reportsto['ID']:
        g.add_vertex(name = str(emp))
    
    for k in range(0, len(reportsto['ID'])):
        for l in range(k + 1, len(reportsto['ID'])):
            if mail_counter[k][l] == 0 or mail_counter[l][k] == 0:
                pass
            else:
                both_sides = mail_counter[k][l] + mail_counter[l][k]
                if both_sides >= threshold:
                    g.add_edge(str(k), str(l))
    
    ndls_ids = list(reportsto[reportsto['ReportsToID'].isin(['former employee account', 'technical email account - not used by employees'])]['ID'])
    g.delete_vertices(ndls_ids) #verices of former emp/tech. acc. deletion
    

    #new random edges loop
    rand_edges_mod = []
    for n in range(N+1):
        g3 = deepcopy(g)
        random_edges = find_random_new_edges(g3, n)
        rand_edges_mod.append(louvain.find_partition(g3, louvain.ModularityVertexPartition).quality())
        
 
    best_edges_mod = []
    #new best edges loop
    for n in range(N+1):
        g4 = deepcopy(g)
        best_edges = find_random_new_edges(g4, n)
        best_edges_mod.append(louvain.find_partition(g4, louvain.ModularityVertexPartition).quality())
        
    
    #main digraph
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
        
    metrics = ["closeness", "betweenness", "indegree", "outdegree", "degree"]
    
    #5%
    IC_5 = []
    for metric in metrics:
        di_g_copy = deepcopy(di_g)
        initial_grp = di_g_copy.vs.select(lambda x:x[metric] > sorted(di_g.vs[metric])[-8])
        IC_5.append((metric, independent_cascades(di_g_copy, initial_grp["name"])))
    
    #10%
    IC_10 = []
    for metric in metrics:
        di_g_copy = deepcopy(di_g)
        initial_grp = di_g_copy.vs.select(lambda x:x[metric] > sorted(di_g.vs[metric])[-16])
        IC_10.append((metric, independent_cascades(di_g_copy, initial_grp["name"])))
        
    result.append(("graf bazowy:", g.summary()))
    result.append(("graf z randomowymi krawędziami", g3.summary()))
    result.append(("graf z najlepszymi krawędziami", g4.summary()))
    result.append(("digraf bazowy:", di_g.summary()))
    result.append(("modularność grafu z randomowymi krawędziami",rand_edges_mod))
    result.append(("modularność grafu z najlepszymi krawędziami", best_edges_mod))
    result.append(("wyniki modelu IC dla 5% inicjalnych węzłów", IC_5))
    result.append(("wyniki modelu IC dla 5% inicjalnych węzłów", IC_10))
    
    txt_res = open("{}.txt".format(datetime.now().strftime("%d-%m-%y %H:%M")), "w")
    for L in result:
        txt_res.write(str(L))
        txt_res.write('\n\n')
    txt_res.close()
    
    return result
    
print('start')
test = wrapper()  
    
    
