import numpy as np
import node2vec
import networkx as nx
import math
import time
import random
import matplotlib.pyplot as plt
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
        print(change)
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


def main():
    #  -----------define G  and  edges_var:
    # parameter: k         preserve: G_var        once good G_var appear, save G_var
    k = 0.6
    nodes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    edges = [(1,2,360),(1,3,240),(2,6,300),(3,4,241),(3,12,242),(4,5,120),(4,11,361),(5,6,239),(5,9,301),(6,8,121),(7,8,180),
             (7,18,122),(8,9,60),(8,16,302),(9,10,181),(10,11,299),(10,15,361),(10,16,238),(10,17,480),(11,12,362),(11,14,243),
             (12,13,182),(13,24,237),(14,15,298),(14,23,244),(15,19,179),(15,22,178),(16,17,119),(16,18,182),(17,19,118),(18,20,244),
             (19,20,245),(20,21,359),(20,22,303),(21,22,123),(21,24,183),(22,23,236),(23,24,124)]
    # edges = [(1,2,6),(1,3,4),(2,6,5),(3,4,4),(3,12,4),(4,5,2),(4,11,6),(5,6,4),(5,9,5),(6,8,2),(7,8,3),
    #          (7,18,2),(8,9,10),(8,16,5),(9,10,3),(10,11,5),(10,15,6),(10,16,4),(10,17,8),(11,12,6),(11,14,4),
    #          (12,13,3),(13,24,4),(14,15,5),(14,23,4),(15,19,3),(15,22,3),(16,17,2),(16,18,3),(17,19,2),(18,20,4),
    #          (19,20,4),(20,21,6),(20,22,5),(21,22,2),(21,24,3),(22,23,4),(23,24,2)]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    edges_sta = [[0 for x in range(25)]for y in range(25)]
    for ele in edges:
        standard = random.uniform(0,k) * ele[2]
        edges_sta[ele[0]][ele[1]] = standard
        edges_sta[ele[1]][ele[0]] = standard
    #  ----------------------G, edges_var -----------------
    # get node_to_node:
    edges_sta = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 125.88628295971701, 128.9278761813104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 125.88628295971701, 0, 0, 0, 0, 163.467915602572, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 128.9278761813104, 0, 0, 121.31844816263772, 0, 0, 0, 0, 0, 0, 0, 99.67617888139037, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 121.31844816263772, 0, 1.724469042794012, 0, 0, 0, 0, 0, 115.37920735257435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1.724469042794012, 0, 9.482547773830495, 0, 0, 132.67993091304135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 163.467915602572, 0, 0, 9.482547773830495, 0, 0, 41.0998259581243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 19.214460753623996, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23.503437939047362, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 41.0998259581243, 19.214460753623996, 0, 28.97186968781234, 0, 0, 0, 0, 0, 0, 14.76683557831601, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 132.67993091304135, 0, 0, 28.97186968781234, 0, 18.751143616714362, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 18.751143616714362, 0, 46.97436866775947, 0, 0, 0, 104.43415204277585, 79.9258954634179, 115.14663293305202, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 115.37920735257435, 0, 0, 0, 0, 0, 46.97436866775947, 0, 152.44540308108483, 0, 71.52765196627014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 99.67617888139037, 0, 0, 0, 0, 0, 0, 0, 152.44540308108483, 0, 91.69810081539137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91.69810081539137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13.399536934437984],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71.52765196627014, 0, 0, 0, 107.86637049936128, 0, 0, 0, 0, 0, 0, 0, 139.7803773552445, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104.43415204277585, 0, 0, 0, 107.86637049936128, 0, 0, 0, 0, 23.836724767871132, 0, 0, 15.308143572040901, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 14.76683557831601, 0, 79.9258954634179, 0, 0, 0, 0, 0, 0, 27.859564453688197, 63.60995055713371, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 115.14663293305202, 0, 0, 0, 0, 0, 27.859564453688197, 0, 0, 4.2368482549450945, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 23.503437939047362, 0, 0, 0, 0, 0, 0, 0, 0, 63.60995055713371, 0, 0, 0, 145.12392406840394, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23.836724767871132, 0, 4.2368482549450945, 0, 0, 71.70606510382882, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 145.12392406840394, 71.70606510382882, 0, 154.3660122291444, 31.887436034504223, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 154.3660122291444, 0, 13.550470816344154, 0, 73.09875321312276],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15.308143572040901, 0, 0, 0, 0, 31.887436034504223, 13.550470816344154, 0, 44.827715042376276, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139.7803773552445, 0, 0, 0, 0, 0, 0, 0, 44.827715042376276, 0, 70.91070495351026],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13.399536934437984, 0, 0, 0, 0, 0, 0, 0, 73.09875321312276, 0, 70.91070495351026, 0]]

    node_to_node = []
    node_to_node.append([])
    for i in range(1,25):
        inn = []
        for j in range(1,25):
            if(edges_sta[i][j] != 0):
                inn.append(j)
        node_to_node.append(inn)
    #  --------get warm start policy:
    p = nx.shortest_path(G,target = 15, weight='weight', method='dijkstra')
    next_node_start = [0 for x in range(25)]
    for i in range(1,25):
        if(i == 15):
            continue
        next_node_start[i] = p.get(i)[1]
    next_node_start = [0,2,6,1,3,4,8,18,7,10,11,4,11,12,15,0,17,19,20,15,21,24,15,14,23]
    # -----------------next_node_start--------------------
    # -------------------main part : choose 2 d and 2 zeta to get error rate every iter:
    # --------intial d1 d2 zeta1 zeta2 get all the evaluate
    d_arr = [8,12,16,22]
    zeta_arr = [1.64]
    Tmax = 50
    error_arr = []
    # ---for each d and zeta combination, we need obtatin a average error rate list[[]]
    for d in d_arr:
        for zeta in zeta_arr:
            next_node_best = best_policy(G,edges_sta,zeta)
            start_evaluate = policy_evaluate(next_node_start,G,edges_sta,zeta)
            best_evaluate = policy_evaluate(next_node_best,G,edges_sta,zeta)
            print(start_evaluate)
            print(best_evaluate)
            error_inn = []
            error_inn.append(assess(start_evaluate,best_evaluate))
            next_node = list(next_node_start)
    #-----------initial---------
            # for m in range(20):
            phi = np.load(file = "/Users/pqh/Desktop/route/Sioux/Sioux_d"+ str(d) + "_" + str(2) + "_phi.npy", allow_pickle = True)
            J = [0 for x in range(25)]
            M = [0 for x in range(25)]
            t = 0
            while (t < Tmax):
                A = np.zeros((23,d+1))
                b = np.zeros(23)
                J = [0 for x in range(25)]
                M = [0 for x in range(25)]
                for i in range(23):
                    start_node = i+1
                    if(start_node >= 15):
                        start_node += 1
                    end_node = next_node[start_node]
                    weight = G.get_edge_data(start_node,end_node).get('weight')
                    b[i] = weight
                    for j in range(d+1):
                        if(end_node != 15):
                            A[i][j] = phi[start_node][j] - phi[end_node][j]
                        else:
                            A[i][j] = phi[start_node][j]
                w_j = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b)
                for i in range(25):
                    if(i == 0 or i == 15):
                        continue
                    J[i] = np.dot(phi[i],w_j)
                    if(J[i] < 0):
                        J[i] = 0
                b1 = np.zeros(23)
                for i in range(23):
                    start_node = i+1
                    if(start_node >= 15):
                        start_node += 1
                    end_node = next_node[start_node]
                    weight = G.get_edge_data(start_node,end_node).get('weight')
                    standard = edges_sta[start_node][end_node]
                    b1[i] = weight**2 + standard**2 + 2 * weight * J[end_node]
                w_m = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b1)
                for i in range(25):
                    if(i == 0 or i == 15):
                        continue
                    M[i] = np.dot(phi[i],w_m)
                    if(M[i] < J[i]**2):
                        M[i] = J[i]**2+ 0.001
                li = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24]
                random.shuffle(li)
                for p in li:
                    ff = 0
                    p_next = next_node[p]
                    p_weight = G.get_edge_data(p,p_next).get('weight')
                    p_sta = edges_sta[p][next_node[p]]
                    p_min = p_weight + J[p_next] + zeta * math.sqrt(p_sta**2 + M[p_next] - J[p_next]**2)
                    neighbor_node = node_to_node[p]
                    for q in neighbor_node:
                        o = q
                        flag = 0
                        f = 0
                        while(f < 25):
                            if(o == 15):
                                flag = 1
                                break
                            if(o == p):
                                break
                            o = next_node[o]
                            f+=1
                        if(flag == 0):
                            continue
                        weight = G.get_edge_data(p,q).get('weight')
                        standard = edges_sta[p][q]
                        p_cur = weight + J[q] + zeta * math.sqrt(standard**2 + M[q] - J[q]**2)
                        if(p_min > p_cur + 0.00001):
                            p_min = p_cur
                            p_next = q
                            ff = 1
                    next_node[p] = p_next
                    if(ff == 1) :
                        break
                # 以上是循环，试试
                cur_evaluate = policy_evaluate(next_node,G,edges_sta,zeta)
                cur_access = assess(cur_evaluate,best_evaluate)
                error_inn.append(cur_access)
                # for ele in error_inn:
                #     if(cur_access < ele + 0.00001 and cur_access > ele - 0.00001):
                #         print(t)
                #         t = Tmax
                #         error_inn.remove(cur_access)
                t+=1
            error_arr.append(error_inn)
    print(error_arr)
    plot1(d_arr, zeta_arr, error_arr,Tmax)
if __name__ == '__main__':
    main()
