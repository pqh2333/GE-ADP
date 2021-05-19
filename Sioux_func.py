"""
author: pqh
date:2021.05.18
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import node2vec
import networkx as nx
import math
import time
import random
import matplotlib.pyplot as plt
import copy
from matplotlib.pyplot import MultipleLocator
# input: cur and best: J + zeta*(M - J^2)^1/2
# return: error rate: (cur-best)/best
def assess(cur: list, best: list):
    sum = 0
    for i in range(len(cur)):
        if(i == 0 or i == 15):
            continue
        a = cur[i] - best[i]
        sum += (a/best[i])
    sum /= 23
    return sum

# input: G and varaince
# return : best_policy
def best_policy(G: nx.Graph(), edges_sta: list, zeta: float):
    p = nx.shortest_path(G,target = 15, weight='weight', method='dijkstra')
    next_node_start = [0 for x in range(25)]
    for i in range(1,25):
        if(i == 15):
            continue
        next_node_start[i] = p.get(i)[1]
    change = 1
    next_node = list(next_node_start)
    node_to_node = []
    node_to_node.append([])
    for i in range(1,25):
        inn = []
        for j in range(1,25):
            if(edges_sta[i][j] != 0):
                inn.append(j)
        node_to_node.append(inn)
    while(change > 0):
        change = 0
        A = np.zeros((46,46))
        b = np.zeros(46)
        for i in range(23):
            start_node = i+1
            if(start_node >= 15):
                start_node += 1
            end_node = next_node[start_node]
            weight = G.get_edge_data(start_node,end_node).get('weight')
            A[i][i] = 1
            A[i+23][i+23] = 1
            b[i] = weight
            b[i+23] = weight**2 + edges_sta[start_node][end_node]**2
            if end_node < 15:
                A[i][end_node-1] = -1
                A[i+23][end_node-1] = -2 * weight
                A[i+23][end_node-1 + 23] = -1
            elif end_node > 15:
                A[i][end_node-2] = -1
                A[i+23][end_node-2] = -2 * weight
                A[i+23][end_node-2 + 23] = -1
        x = np.dot(np.linalg.inv(A),b.T)
        for i in range(25):
            if(i == 0 or i == 15):
                continue
            inn = node_to_node[i]
            cur_node_next = next_node[i]
            cur_min = 0
            weight = G.get_edge_data(i,cur_node_next).get('weight')
            if(cur_node_next == 15):
                cur_min = weight + zeta * math.sqrt(edges_sta[i][cur_node_next]**2)
            elif(cur_node_next < 15):
                cur_min = weight + x[cur_node_next-1] + zeta * math.sqrt(edges_sta[i][cur_node_next]**2 + x[cur_node_next-1 + 23] - x[cur_node_next-1]**2)
            else:
                cur_min = weight + x[cur_node_next-2] + zeta * math.sqrt(edges_sta[i][cur_node_next]**2 + x[cur_node_next-2 + 23] - x[cur_node_next-2]**2)
            for end in inn:
                weight1 = G.get_edge_data(i,end).get('weight')
                cur = 0
                if(end == 15):
                    cur = weight1 + zeta * math.sqrt(edges_sta[i][end]**2)
                elif(end < 15):
                    cur = weight1 + x[end-1] + zeta * math.sqrt(edges_sta[i][end]**2 + x[end-1 + 23] - x[end-1]**2)
                else:
                    cur = weight1 + x[end-2] + zeta * math.sqrt(edges_sta[i][end]**2 + x[end-2 + 23] - x[end-2]**2)
                if(cur < cur_min -0.00000001):
                    cur_min = cur
                    next_node[i] = end
                    change += 1
        # print(change)
    return next_node
# input: policy,G and varaince
# output: cur
def policy_evaluate(next_node: list , G: nx.Graph(), edges_sta: list, zeta: float):
    A = np.zeros((46,46))
    b = np.zeros(46)
    for i in range(23):
        start_node = i+1
        if(start_node >= 15):
            start_node += 1
        end_node = next_node[start_node]
        weight = G.get_edge_data(start_node,end_node).get('weight')
        A[i][i] = 1
        A[i+23][i+23] = 1
        b[i] = weight
        b[i+23] = weight**2 + edges_sta[start_node][end_node]**2
        if end_node < 15:
            A[i][end_node-1] = -1
            A[i+23][end_node-1] = -2 * weight
            A[i+23][end_node-1 + 23] = -1
        elif end_node > 15:
            A[i][end_node-2] = -1
            A[i+23][end_node-2] = -2 * weight
            A[i+23][end_node-2 + 23] = -1
    x = np.dot(np.linalg.inv(A),b.T)
    cur = [0 for x in range(25)]
    k = 0
    for i in range(25):
        if i == 0 or i == 15:
            continue
        cur[i] = x[k] + zeta * math.sqrt(x[k+23]-x[k]**2)
        k += 1
    return cur
# input: d,zeta and error rate
def plot1(d_arr: list, zeta_arr: list, error_arr: list, Tmax: int):
    plt.figure(figsize=(12, 8))
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 35,
    }
    font3 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
    }
    x = np.arange(Tmax)
    k = 0
    l = []
    for d in d_arr:
        for zeta in zeta_arr:
            x = np.arange(len(error_arr[k]))
            if (k == 0) :
                plt.plot(x,error_arr[k],'--',color = 'k',linewidth = 3)
            if (k == 1) :
                plt.plot(x,error_arr[k],'-',color = 'g',linewidth = 3)
            if (k == 2) :
                plt.plot(x,error_arr[k],'-.',color = 'b',linewidth = 3)
            if (k == 3):
                plt.plot(x,error_arr[k],':',color = 'r',linewidth = 3)
            l.append("d = "+str(d))
            k+=1
    plt.legend(l,loc = 'upper right',prop = font3)
    plt.xlabel('Iteration',font2)
    plt.ylabel('Relative error',font2)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim(0,50)
    plt.show()

def plot2(d_arr:list, error_ave:list, time_ave:list):
    x = d_arr
    y1 = error_ave
    y2 = time_ave
    fig = plt.figure()
    plt.title('Error rate and Efficiency')
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1)
    ax1.set_ylabel('Error rate')
    ax1.set_title("Error rate and Efficiency")

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, 'r')
    ax2.set_xlim([0,24])
    ax2.set_ylabel('Runtime')
    # ax2.set_xlabel('Same X for both exp(-x) and ln(x)')
    plt.show()
# def random_start(node_to_node: list):
def plot3(d_arr:list, error_ave: list, time_ave:list, error_sta:list, time_sta:list):
    x = d_arr
    y1 = np.array(error_ave)
    y2 = np.array(time_ave)
    error_sta = np.array(error_sta)
    time_sta = np.array(time_sta)
    fig = plt.figure()
    plt.title('Error rate and Efficiency')
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1,'b',)
    ax1.set_ylabel('Error rate')
    ax1.set_title("Error rate and Efficiency")
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, 'r')
    ax2.set_xlim([0,24])
    ax2.set_ylabel('Runtime')
    ax1.plot(x,y1+error_sta)
    plt.fill_between(x, y1-error_sta, y1+error_sta, color = 'cornflowerblue')
    plt.fill_between(x, y2-time_sta, y2+time_sta, color = 'lightcoral')
    plt.show()

def plot4(error_ave: list, time_ave: list, error_sta: list, time_sta: list):
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 35,
    }
    font3 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
    }
    d_arr = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    x = d_arr
    y1 = np.array(error_ave)
    y2 = np.array(time_ave)
    error_sta = np.array(error_sta)
    time_sta = np.array(time_sta)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(x, y1,color = 'b',label ='relative error')
    plt.yticks(size =25)
    plt.xticks(size =25)
    plt.xlabel('Dimensionality (d)',font2)
    plt.errorbar(d_arr,y1,color = 'b', yerr =error_sta,capsize=4)
    ax1.set_ylabel('Relavtive error',font2)
    ax2 = ax1.twinx()
    ax2.set_xlim(left = 3, right = 23)
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    plt.xticks(np.arange(3,25,2))
    ax2.set_ylabel('Runtime (s)',font2)
    l2 = ax2.plot(x, y2, ':',color ='r',label = 'run time',linewidth = 3)
    fig.legend(loc= 1 ,bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, prop = font2)
    plt.ylim(0.1,0.55)
    plt.yticks(np.arange(0.15,0.55,0.05))
    plt.yticks(size =25)
    plt.show()
