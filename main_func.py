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
import os



def wrapper(threshold = 20, N = 50, groups = 5, step = 1):
    start_time = datetime.now().strftime("%d-%m-%y %H:%M")
    
    if not os.path.exists('/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/Wyniki/{}'.format(start_time)):
        os.makedirs('/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/Wyniki/{}'.format(start_time))
    
    result = [('data', start_time), ("threshold", threshold), ("Max nowych krawędzi", N), ("Ile krawędzi jest dodawanych w jednej iteracji:", groups)]
    
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
    for n in range(0, N+1, step):
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
        if type(L[1]) == ig.Graph:
            txt_res.write(L[0], str(L[1].summary()))
            txt_res.write('\n\n')
        elif ig.VertexSeq in [type(x) for x in L[1]]:
            txt_res.write(L[0], [x['name'] for x in L[1]])
            txt_res.write('\n\n')
        else:
            txt_res.write(str(L))
            txt_res.write('\n\n')
    txt_res.close()
    
    ###########
    ## plots ##
    ###########
    
    #styles
    sns.set_style('ticks')
    
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
    
    # # # # # # # #
    #graphs' plots
    
    #base graph (g)
    ig.plot(g, **visual_style, mark_groups = True).save(Wyniki/{}/base_graph.png'.format(start_time))
                                                        
    partition = louvain.find_partition(g, louvain.ModularityVertexPartition)
    ig.plot(partition, **visual_style, mark_groups = True).save(Wyniki/{}/partition_base_graph.png'.format(start_time))
                                                        
    #improved graph (g4)
    partition = louvain.find_partition(g4, louvain.ModularityVertexPartition)
    ig.plot(g4, **visual_style, mark_groups = True).save(Wyniki/{}/partition_improved_graph.png'.format(start_time))
    
    #activation graph (di_g) (?)
    
    # # # # # # # # 3 #
    #dataframe modularity
    modularities = pd.DataFrame(data = [list(range(len(rand_edges_mod[1]))) + list(range(len(rand_edges_mod[1]))), rand_edges_mod[1] + best_edges_mod[1], ["random"]*len(rand_edges_mod[1]) + ["best"] * len(best_edges_mod[1])], index = ["index", "modularity", "type"]).T
    modularities.index = modularities.index.astype(int)
    modularities.modularity = modularities.modularity.astype(float)
    
    #modularity plot
    fig, ax = plt.subplots()

    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "modularity", data = modularities, hue = "type", ax=ax)
    sns.despine() #leaves only bottom and left axis

    fig.savefig('Wyniki/{}/modularity_plot.png'.format(start_time))
    
    #spread iterations no. dataframe
    spread = pd.DataFrame(data = [list(range(N+1)) * len(test[13][1]), [y for x in test[13][1] for y in x[1]], [x[0] for x in test[13][1] for _ in range(N+1)]], index = ["index", "spread_iter", "type"]).T
    spread.index = spread.index.astype(int)
    spread.spread_iter = spread.spread_iter.astype(int)
    
    #spread iterations no. plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_iter", data = spread, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/spread_iterations.png'.format(start_time))
    
    
    #spread range dataframe
    spread2 = pd.DataFrame(data = [list(range(N+1)) * len(test[13][1]), [y for x in test[13][1] for y in x[2]], [x[0] for x in test[13][1] for _ in range(N+1)]], index = ["index", "spread_range", "type"]).T
    spread2.index = spread2.index.astype(int)
    spread2.spread_range = spread2.spread_range.astype(int)
    
    #spread range plot
     fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_range", data = spread2, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/spread_activations.png'.format(start_time))
    
    
    return result
###############################################################################


print('start')
test = wrapper(20, 5, 5, 1)
    
N = test[2][1]

#spread iterations no. dataframe
spread = pd.DataFrame(data = [list(range(N+1)) * len(test[13][1]), [y for x in test[13][1] for y in x[1]], [x[0] for x in test[13][1] for _ in range(N+1)]], index = ["index", "spread_iter", "type"]).T
spread.index = spread.index.astype(int)
spread.spread_iter = spread.spread_iter.astype(int)

#spread iterations no. plot
#sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.set_style('ticks')


fig, ax = plt.subplots()
fig.set_size_inches(23.4, 16.54)
sns.lineplot(x = "index", y = "spread_iter", data = spread, hue = "type", ax=ax)
sns.despine()
fig.savefig('Wyniki/{}/0aa.png'.format(start_time))

#spread range dataframe
spread2 = pd.DataFrame(data = [list(range(N+1)) * len(test[13][1]), [y for x in test[13][1] for y in x[2]], [x[0] for x in test[13][1] for _ in range(N+1)]], index = ["index", "spread_range", "type"]).T
spread2.index = spread2.index.astype(int)
spread2.spread_range = spread2.spread_range.astype(int)

#spread range plot
plt.figure(figsize = (25, 15))
sns.relplot(x = "index", y = "spread_range", kind = "line", data = spread2, hue = "type")
         