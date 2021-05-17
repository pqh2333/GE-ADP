import numpy as np
import node2vec
import networkx as nx
import openpyxl
import time
def main():
    nod = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    edg = [(1,2),(1,3),(2,6),(3,4),(3,12),(4,5),(4,11),(5,6),(5,9),(6,8),(7,8),(7,18),(8,9),(8,16),(9,10),(10,11),(10,15),(10,16),(10,17),(11,12),(11,14),
             (12,13),(13,24),(14,15),(14,23),(15,19),(15,22),(16,17),(16,18),(17,19),(18,20),(19,20),(20,21),(20,22),(21,22),(21,24),(22,23),(23,24)]
    G1 = nx.Graph()
    G1.add_nodes_from(nod)
    G1.add_edges_from(edg)
    for k in range(1000,2000):
        for m in range (22,1,-1):
            print(time.time())
            initil = node2vec.Node2Vec(G1,dimensions= m,walk_length = 50, num_walks = 60, p = 2, q= 0.5)
            print(time.time())
            model = initil.fit()
            print(model.wv.vectors)
            phi = []
            phi.append([])
            for i in range(24):
                phi_i = model.wv.get_vector(str(i+1)).tolist()
                phi_i.append(1)
                phi.append(phi_i)
            np.save(file = "/Users/pqh/Desktop/route/Sioux/Sioux_d"+ str(m) +"_" + str(k) + "_phi.npy", arr = phi)
            print(time.time())
if __name__ == '__main__':
    main()
