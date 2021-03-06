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
from collections.abc import Iterable



def wrapper(threshold = 20, N = 100, groups = 5, step = 1, IC_loop = 1000):
    start_time = datetime.now().strftime("%d-%m-%y %H:%M")
    
    if not os.path.exists('/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/Wyniki/{}'.format(start_time)):
        os.makedirs('/home/barni13/Desktop/Studia/Semestr 7/Praca Dyplomowa/Wyniki/{}'.format(start_time))
    
    result = [('data', start_time), ("threshold", threshold), ("N = max nowych krawędzi", N), ("Ile krawędzi jest dodawanych w jednej iteracji:", groups), ("Co ile krawędzi iterujemy od 0 do N:", step), ("Liczba iteracji w modelu IC:", IC_loop)]
    
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
    
    list_both_sides = []
    
    for k in range(0, len(reportsto['ID'])):
        for l in range(k + 1, len(reportsto['ID'])):
            if mail_counter[k][l] == 0 or mail_counter[l][k] == 0:
                pass
            else:
                both_sides = mail_counter[k][l] + mail_counter[l][k]
                list_both_sides.append(both_sides)
                if both_sides >= threshold:
                    g.add_edge(str(k), str(l), added = 0)
    
    ndls_ids = list(reportsto[reportsto['ReportsToID'].isin(['former employee account', 'technical email account - not used by employees'])]['ID'])
    g.delete_vertices(ndls_ids) #verices of former emp/tech. acc. deletion
        
    metrics = ["closeness", "betweenness", "indegree", "outdegree", "degree"]
    
    rand_edges_mod = []
    best_edges_mod = []
    
    
    IC_5_new = [("closeness", [], []), ("betweenness", [], []), ("indegree", [], []), ("outdegree", [], []), ("degree", [], [])]
    IC_10_new = [("closeness", [], []), ("betweenness", [], []), ("indegree", [], []), ("outdegree", [], []), ("degree", [], [])]
    
    #IC_5_all = [("closeness", [], []), ("betweenness", [], []), ("indegree", [], []), ("outdegree", [], []), ("degree", [], [])]
    #IC_10_all = [("closeness", [], []), ("betweenness", [], []), ("indegree", [], []), ("outdegree", [], []), ("degree", [], [])]
    
    #one loop
    for n in range(0, N+1, step):
        if n != 0:
            print(datetime.now().strftime("%d-%m-%y %H:%M"), '\n Ukończono: '+str((n - step)/(N)*100)+'%')
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
            di_g.add_edge(x.tuple[0], x.tuple[1], weights = weights[x.tuple[0]][x.tuple[1]], added = x["added"])
            di_g.add_edge(x.tuple[1], x.tuple[0], weights = weights[x.tuple[1]][x.tuple[0]], added = x["added"])
        
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
            #new
            tmp_res_new_1 = []
            tmp_res_new_2 = []
            #all
            #tmp_res_all_1 = []
            #tmp_res_all_2 = []
            for j in range(IC_loop):    
                #new
                ic5_res_new = independent_cascades_new(di_g_copy, initial_grp_5[i]["name"])
                tmp_res_new_1.append(len(ic5_res_new[0]))
                tmp_res_new_2.append(len(ic5_res_new[1]))            
                #all
                #ic5_res_all = independent_cascades_all(di_g_copy, initial_grp_5[i]["name"])
                #tmp_res_all_1.append(len(ic5_res_all[0]))
                #tmp_res_all_2.append(len(ic5_res_all[1]))
                
            IC_5_new[i][1].append(sum(tmp_res_new_1)/len(tmp_res_new_1))
            IC_5_new[i][2].append(sum(tmp_res_new_2)/len(tmp_res_new_2))
            #IC_5_all[i][1].append(sum(tmp_res_all_1)/len(tmp_res_all_1))
            #IC_5_all[i][2].append(sum(tmp_res_all_2)/len(tmp_res_all_2))
    
        if n == 0:
            initial_grp_10 = []
        #10%
        for i in range(len(metrics)):
            di_g_copy = deepcopy(di_g)
            #selecting initial_grps
            if n == 0:  
                initial_grp_10.append(di_g_copy.vs.select(lambda x:x[metrics[i]] > sorted(di_g.vs[metrics[i]])[-16]))
            #new
            tmp_res_new_3 = []
            tmp_res_new_4 = []
            #all
            #tmp_res_all_3 = []
            #tmp_res_all_4 = []
            for j in range(IC_loop):
                 #new
                ic10_res_new = independent_cascades_new(di_g_copy, initial_grp_10[i]["name"])
                tmp_res_new_3.append(len(ic10_res_new[0]))
                tmp_res_new_4.append(len(ic10_res_new[1]))            
                #all
                #ic10_res_all = independent_cascades_all(di_g_copy, initial_grp_10[i]["name"])
                #tmp_res_all_3.append(len(ic10_res_all[0]))
                #tmp_res_all_4.append(len(ic10_res_all[1]))
                
            IC_10_new[i][1].append(sum(tmp_res_new_3)/len(tmp_res_new_3))
            IC_10_new[i][2].append(sum(tmp_res_new_4)/len(tmp_res_new_4))
            #IC_10_all[i][1].append(sum(tmp_res_all_3)/len(tmp_res_all_3))
            #IC_10_all[i][2].append(sum(tmp_res_all_4)/len(tmp_res_all_4))
        
    result.append(("graf bazowy:", g))
    result.append(("graf z randomowymi krawędziami", g3))
    result.append(("graf z najlepszymi krawędziami", g4))
    result.append(("digraf bazowy:", di_g))
    result.append(("modularność grafu z randomowymi krawędziami",rand_edges_mod))
    result.append(("modularność grafu z najlepszymi krawędziami", best_edges_mod))
    result.append(("Miary:", metrics))
    result.append(("Grupa inizializująca dla 5% próby", initial_grp_5))
    result.append(("Grupa inizializująca dla 10% próby", initial_grp_10))
    result.append(("wyniki modelu IC dla 5% inicjalnych węzłów (nowe)", IC_5_new))
    result.append(("wyniki modelu IC dla 10% inicjalnych węzłów (nowe)", IC_10_new))
    #result.append(("wyniki modelu IC dla 5% inicjalnych węzłów (wszystkie)", IC_5_all))
    #result.append(("wyniki modelu IC dla 10% inicjalnych węzłów (wszystkiue)", IC_10_all))
    
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
    ig.plot(g, **visual_style, mark_groups = True).save("Wyniki/{}/base_graph.png".format(start_time))
                                                        
    partition = louvain.find_partition(g, louvain.ModularityVertexPartition)
    ig.plot(partition, **visual_style, mark_groups = True).save("Wyniki/{}/partition_base_graph.png".format(start_time))
                                                        
    #improved graph (g4)
    partition2 = louvain.find_partition(g4, louvain.ModularityVertexPartition)
    ig.plot(partition2, **visual_style, mark_groups = True).save("Wyniki/{}/partition_improved_graph.png".format(start_time))
    
    #activation graph (di_g) (?)
    
    # # # # # # # # # #
    #threshold plots
    thresholds = []
    above_trsh = []
    
    for threshold in np.linspace(min(list_both_sides), max(list_both_sides), 2000):
        thresholds.append(threshold)
        above_trsh.append(len([x for x in list_both_sides if x >= threshold]))
    
    df_thresholds = pd.DataFrame(data = [thresholds, above_trsh, [x / len(g.vs['name']) for x in above_trsh]], index = ["thresholds", "above_trsh", "ratio"]).T
    h_line = pd.DataFrame(data = [[0, 7], [10, 7.001]], columns = ['x', 'y'])
    v_line = pd.DataFrame(data = [[10, 0], [10.00001, 7.01]], columns = ['x', 'y'])
    
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    #sns.lineplot(x = "thresholds", y ="ratio", data = df_thresholds, ax = ax[0])
    #sns.despine()
    sns.lineplot(x = "thresholds", y ="ratio", data = df_thresholds[0:40], ax = ax[1])
    sns.lineplot(x = 'x', y = 'y', data = v_line, ax = ax[1], linewidth = 1.15, color = 'grey', alpha=.75)
    sns.lineplot(x = 'x', y = 'y', data = h_line, ax = ax[1], linewidth = 1.15, color = 'grey', alpha=.75)
    sns.despine()
    #ax[0].set_title('Wykres całości', size = 28)
    #ax[1].set_title('Wykres dla najmniejszych wartości poziomów progowych', size = 28)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    #ax[0].set_ylabel('Gęstość grafu', fontsize = 28)
    #ax[0].set_xlabel('Poziom progowy', fontsize = 28)
    ax.set_ylabel('Gęstość grafu', fontsize = 28)
    ax.set_xlabel('Poziom progowy', fontsize = 28)
    ax[0].tick_params(axis='both', which='major', labelsize=24)
    #ax.tick_params(axis='both', which='major', labelsize=24)
    fig.savefig('Wyniki/{}/thresholds_subplot5.png'.format(start_time))
    
    
    #dataframe modularity
    modularities = pd.DataFrame(data = [list(range(0, N+1, step)) * 2, rand_edges_mod + best_edges_mod, ["random"]*len(rand_edges_mod) + ["best"] * len(best_edges_mod)], index = ["index", "modularity", "type"]).T
    modularities.index = modularities.index.astype(int)
    modularities.modularity = modularities.modularity.astype(float)
    
    #modularity plot
    fig, ax = plt.subplots()

    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "modularity", data = modularities, hue = "type", ax=ax)
    sns.despine() #leaves only bottom and left axis

    fig.savefig('Wyniki/{}/modularity_plot.png'.format(start_time))
    
    #NEW
    
    #IC_5_new
    #spread iterations no. dataframe
    spread = pd.DataFrame(data = [list(range(0, N+1, step)) * len(IC_5_new), [y for x in IC_5_new for y in x[1]], [x[0] for x in IC_5_new for _ in range(0, N+1, step)]], index = ["index", "spread_iter", "type"]).T
    spread.index = spread.index.astype(int)
    spread.spread_iter = spread.spread_iter.astype(float)
    
    #spread iterations no. plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_iter", data = spread, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC5_new_spread_iterations.png'.format(start_time))
    
    
    #spread range dataframe
    spread2 = pd.DataFrame(data = [list(range(0, N+1, step)) * len(IC_5_new), [y for x in IC_5_new for y in x[2]], [x[0] for x in IC_5_new for _ in range(0, N+1, step)]], index = ["index", "spread_range", "type"]).T
    spread2.index = spread2.index.astype(int)
    spread2.spread_range = spread2.spread_range.astype(float)
    
    #spread range plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_range", data = spread2, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC5_new_spread_activations.png'.format(start_time))
    
    #IC_10_new
    #spread iterations no. dataframe
    spread = pd.DataFrame(data = [list(range(0, N+1, step)) * len(IC_10_new), [y for x in IC_10_new for y in x[1]], [x[0] for x in IC_10_new for _ in range(0, N+1, step)]], index = ["index", "spread_iter", "type"]).T
    spread.index = spread.index.astype(int)
    spread.spread_iter = spread.spread_iter.astype(float)
    
    #spread iterations no. plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_iter", data = spread, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC10_new_spread_iterations.png'.format(start_time))
    
    
    #spread range dataframe
    spread2 = pd.DataFrame(data = [list(range(0, N+1, step)) * len(IC_10_new), [y for x in IC_10_new for y in x[2]], [x[0] for x in IC_10_new for _ in range(0, N+1, step)]], index = ["index", "spread_range", "type"]).T
    spread2.index = spread2.index.astype(int)
    spread2.spread_range = spread2.spread_range.astype(float)
    
    #spread range plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_range", data = spread2, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC10_new_spread_activations.png'.format(start_time))
    
    #spread vs mod.
    spread_n = [x[0] for x in IC_10_new] 
    spread_v = [x[2] for x in IC_10_new]
    modularity = [best_edges_mod]
    sm = pd.DataFrame( data = spread_v + modularity, index = spread_n + ['modularity']).T
    sm = sm.melt('modularity', var_name='cols', value_name='vals')

    
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.scatterplot(x="modularity", y="vals", hue='cols', data=sm, ax = ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC10_new_spread_vs_mod'.format(start_time))
    
    #ALL
    """
    #IC_5_all
    #spread iterations no. dataframe
    spread = pd.DataFrame(data = [list(range(0, N+1, step)) * len(IC_5_all), [y for x in IC_5_all for y in x[1]], [x[0] for x in IC_5_all for _ in range(0, N+1, step)]], index = ["index", "spread_iter", "type"]).T
    spread.index = spread.index.astype(int)
    spread.spread_iter = spread.spread_iter.astype(float)
    
    #spread iterations no. plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_iter", data = spread, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC5_all_spread_iterations.png'.format(start_time))
    
    
    #spread range dataframe
    spread2 = pd.DataFrame(data = [list(range(0, N+1, step)) * len(IC_5_all), [y for x in IC_5_all for y in x[2]], [x[0] for x in IC_5_all for _ in range(0, N+1, step)]], index = ["index", "spread_range", "type"]).T
    spread2.index = spread2.index.astype(int)
    spread2.spread_range = spread2.spread_range.astype(float)
    
    #spread range plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_range", data = spread2, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC5_all_spread_activations.png'.format(start_time))
    
    #IC_10_all
    #spread iterations no. dataframe
    spread = pd.DataFrame(data = [list(range(0, N+1, step)) * len(IC_10_all), [y for x in IC_10_all for y in x[1]], [x[0] for x in IC_10_all for _ in range(0, N+1, step)]], index = ["index", "spread_iter", "type"]).T
    spread.index = spread.index.astype(int)
    spread.spread_iter = spread.spread_iter.astype(float)
    
    #spread iterations no. plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_iter", data = spread, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC10_all_spread_iterations.png'.format(start_time))
    
    
    #spread range dataframe
    spread2 = pd.DataFrame(data = [list(range(0, N+1, step)) * len(IC_10_all), [y for x in IC_10_all for y in x[2]], [x[0] for x in IC_10_all for _ in range(0, N+1, step)]], index = ["index", "spread_range", "type"]).T
    spread2.index = spread2.index.astype(int)
    spread2.spread_range = spread2.spread_range.astype(float)
    
    #spread range plot
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.lineplot(x = "index", y = "spread_range", data = spread2, hue = "type", ax=ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC10_all_spread_activations.png'.format(start_time))
    
    #spread vs mod.
    spread_n = [x[0] for x in IC_10_all] 
    spread_v = [x[2] for x in IC_10_all]
    modularity = [best_edges_mod]
    sm = pd.DataFrame( data = spread_v + modularity, index = spread_n + ['modularity']).T
    sm = sm.melt('modularity', var_name='cols', value_name='vals')

    
    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 16.54)
    sns.scatterplot(x="modularity", y="vals", hue='cols', data=sm, ax = ax)
    sns.despine()
    fig.savefig('Wyniki/{}/IC10_all_spread_vs_mod'.format(start_time))
    """
   
    txt_res = open("Wyniki/{}/wynik_badań.txt".format(start_time), "w")
    for L in result:
        if isinstance(L[1], Iterable):
            if ig.VertexSeq in [type(x) for x in L[1]]:
                txt_res.write(L[0])
                for x in L[1]:
                    txt_res.write(str(x['name']))
                txt_res.write('\n\n')
            else:
                txt_res.write(str(L))
                txt_res.write('\n\n')
        else:    
            if type(L[1]) == ig.Graph:
                txt_res.write(L[0])
                txt_res.write(str(L[1].summary()))
                txt_res.write('\n\n')
            else:
                txt_res.write(str(L))
                txt_res.write('\n\n')
    txt_res.close()
    
    return result
###############################################################################


print('start')
output_result = wrapper(threshold = 10, N = 250, groups = 10, step = 25, IC_loop = 1000)

"""
spread_n = [x[0] for x in output_result[14][1]] #output_result[14][1] == IC_10
spread_v = [x[2] for x in output_result[14][1]]
modularity = [output_result[9][1]] #output_result[9][1] == best_edges_mod

sm = pd.DataFrame( data = spread + modularity, index = spread_n + ['modularity']).T

plt.figure()
for i in spread_n:
    sns.scatterplot(x="modularity", y = i, data=sm)
#sm = sm.melt('modularity', var_name='cols', value_name='vals')
#sns.scatterplot(x="modularity", y="vals", hue='cols', data=sm)
"""