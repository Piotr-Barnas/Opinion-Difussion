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
import seaborn as sns
from needed_functions import *              #find_best_new_edges, find_random_new_edges, independent_cascades


def wrapper(threshold = 20, N = 50, groups = 5):
    result = [('data', datetime.now().strftime("%d-%m-%y %H:%M")), ("threshold", threshold), ("Max nowych krawędzi", N), ("Ile krawędzi jest dodawanych w jednej iteracji:", groups)]
    
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
        
    metrics = ["closeness", "betweenness", "indegree", "outdegree", "degree"]
    
    rand_edges_mod = []
    best_edges_mod = []
    
    IC_5 = [("closeness", [], []), ("betweenness", [], []), ("indegree", [], []), ("outdegree", [], []), ("degree", [], [])]
    IC_10 = [("closeness", [], []), ("betweenness", [], []), ("indegree", [], []), ("outdegree", [], []), ("degree", [], [])]
    
    #one loop
    for n in range(N+1):
        if n != 0:
            print(datetime.now().strftime("%d-%m-%y %H:%M"), '\n Ukończono: '+str((n - 1)/N*100)+'%')
        g3 = deepcopy(g)
        random_edges = find_random_new_edges(g3, n)
        rand_edges_mod.append(louvain.find_partition(g3, louvain.ModularityVertexPartition).quality())

        g4 = deepcopy(g)
        best_edges = find_best_new_edges(g4, n, groups)
        best_edges_mod.append(louvain.find_partition(g4, louvain.ModularityVertexPartition).quality())
        
        #main digraph creation
        di_g = ig.Graph(directed = True)
        di_g.add_vertices(list(g4.vs["name"]))
        
        for x in g4.es:
            di_g.add_edge(x.tuple[0], x.tuple[1], weights = weights[x.tuple[0]][x.tuple[1]])
            di_g.add_edge(x.tuple[1], x.tuple[0], weights = weights[x.tuple[1]][x.tuple[0]])
        
        di_g.vs["closeness"] = di_g.closeness()
        di_g.vs["betweenness"] = di_g.betweenness() 
        #di_g.vs["clustering_coef] = ...
        di_g.vs["indegree"] = di_g.indegree()
        di_g.vs["outdegree"] = di_g.outdegree()
        di_g.vs["degree"] = di_g.degree()
        
        if n == 0:
            initial_grp_5 = []
        #5%
        for i in range(len(metrics)):
            di_g_copy = deepcopy(di_g)
            #selecting initial_grps
            if n == 0:
                initial_grp_5.append(di_g_copy.vs.select(lambda x:x[metrics[i]] > sorted(di_g.vs[metrics[i]])[-8]))
            ic5_res = independent_cascades(di_g_copy, initial_grp_5[i]["name"])
            IC_5[i][1].append(len(ic5_res[0]))
            IC_5[i][2].append(len(ic5_res[1]))
        
        if n == 0:
            initial_grp_10 = []
        #10%
        for i in range(len(metrics)):
            di_g_copy = deepcopy(di_g)
            #selecting initial_grps
            if n == 0:  
                initial_grp_10.append(di_g_copy.vs.select(lambda x:x[metrics[i]] > sorted(di_g.vs[metrics[i]])[-16]))
            ic10_res = independent_cascades(di_g_copy, initial_grp_10[i]["name"])
            IC_10[i][1].append(len(ic10_res[0]))
            IC_10[i][2].append(len(ic10_res[1]))
        
    result.append(("graf bazowy:", g))
    result.append(("graf z randomowymi krawędziami", g3))
    result.append(("graf z najlepszymi krawędziami", g4))
    result.append(("digraf bazowy:", di_g))
    result.append(("modularność grafu z randomowymi krawędziami",rand_edges_mod))
    result.append(("modularność grafu z najlepszymi krawędziami", best_edges_mod))
    result.append(("Miary:", metrics))
    result.append(("Grupa inizializująca dla 5% próby", initial_grp_5))
    result.append(("Grupa inizializująca dla 10% próby", initial_grp_10))
    result.append(("wyniki modelu IC dla 5% inicjalnych węzłów", IC_5))
    result.append(("wyniki modelu IC dla 10% inicjalnych węzłów", IC_10))
    
    txt_res = open("Wyniki/{}.txt".format(datetime.now().strftime("%d-%m-%y %H:%M")), "w")
    for L in result:
        if type(L) == ig.Graph:
            txt_res.write(str(L.summary()))
            txt_res.write('\n\n')
        else:
            txt_res.write(str(L))
            txt_res.write('\n\n')
    txt_res.close()
    
    return result
    
print('start')
test = wrapper(20, 2, 2)
    
N = test[2][1]
random_edges_mod = test[8][1]
best_edges_mod = test[9][1]

modularities = pd.DataFrame(data = [test[8][1], test[9][1]], index = ["random_mod", "best_mod"]).T.reset_index()

sns.set_style("whitegrid")
plt.figure()
#plt.plot(range(N+1), random_edges_mod, range(N+1), best_edges_mod)
sns.relplot(x = "index", y = "random_mod", data = modularities, kind = "line")

"""
metrics = ["closeness", "betweenness", "indegree", "outdegree", "degree"]

plt.figure()
for metric in metrics:
    activated = []
    for res in test[11][1]:
        if res[0] == metric:
            activated.append(res[1])
    plt.plot(range(len(activated)), activated)
plt.legend(metrics)  
"""            