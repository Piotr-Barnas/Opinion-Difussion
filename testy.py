#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:49:43 2020

@author: barni13
"""

import pandas as pd
import igraph as ig
import numpy as np
import leidenalg
import itertools as itl
from copy import deepcopy


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

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


ndls_ids = list(reportsto[reportsto['ReportsToID'].isin(['former employee account', 'technical email account - not used by employees'])]['ID']) #needless email accounts' IDs
act_emp_rep = reportsto[~reportsto['ID'].isin(ndls_ids)] #"reportsto" containing only active employees 
act_emp_mails = communication[communication['Sender'].isin(ndls_ids).apply(int) + communication['Recipient'].isin(ndls_ids).apply(int) == 0] 


mail_counter = np.zeros((len(reportsto['ID']), len(reportsto['ID'])))

g = ig.Graph(len(reportsto['ID']))
for i in range(0, len(communication)):
    mail_counter[communication.iloc[i]['Sender']][communication.iloc[i]['Recipient']] =+ 1

weights = mail_counter/mail_counter.sum(axis=1, keepdims=True)
weights = np.nan_to_num(weights)

for k in range(0, len(reportsto['ID'])):
    for l in range(0, len(reportsto['ID'])):
        if k == l:
            pass
        if weights[k][l] != 0:
            g.add_edge(k, l, weights = weights[k][l])

g.delete_vertices(ndls_ids)

layout = g.layout("tree") #tree / grid_fr      kk (?)
visual_style = {}
visual_style["vertex_size"] = 20
#visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
#visual_style["vertex_label"] = g.vs["name"]
#visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
visual_style["layout"] = layout
visual_style["bbox"] = (1200, 1200)
visual_style["margin"] = 20
ig.plot(g, **visual_style).show()

partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
print(partition.modularity)

ig.plot(partition, **visual_style).show()

pairs = list(itl.permutations(act_emp_rep['ID'][0:10], 2))

for m in range (len(pairs)):
    g_new = deepcopy(g)
    g_new.add_edge(pairs[m][0], pairs[m][1], weights = np.mean(g.es['weights']))
    print(leidenalg.find_partition(g_new, leidenalg.ModularityVertexPartition).modularity)

