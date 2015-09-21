import numpy as np
import scipy as sp
import scipy.spatial.distance as dist
import networkx as nx

def eps_graph(X, eps=1):
    Xdist = (dist.squareform(dist.pdist(X)) < eps).astype(int)
    np.fill_diagonal(Xdist, 0)
    #print(Xdist.min(), Xdist.max())

    return Xdist

def modularity(G, P, L, m):
    Lr = L.reshape((1, L.shape[0]))
    L1 = Lr.repeat(L.shape[0], axis=0)
    L2 = Lr.T.repeat(L.shape[0], axis=1)

    return (1 / (2 * m) * ((G - P) * (L1 == L2))[:, 0]).sum()

def process(G, max_loops=10, min_modularity=.0001):
    Gnode = np.arange(G.shape[0])
    node_shuffle = Gnode.copy()
    Glabel = Gnode.copy()
    Gdeg = G.sum(axis=0)
    m = G.sum() / 2
    GP = np.outer(Gdeg, Gdeg) / (2 * m)
    loop = 0
    modular_cur = modularity(G, GP, Glabel, m)
    modular_gain = 1000.0

    while loop < max_loops and modular_gain > min_modularity:
        #print('Loop', loop)
        np.random.shuffle(node_shuffle)
        
        for n in node_shuffle:
            neighbor = np.where(G[:, n] == 1)[0]
            neighbor_label = Glabel[neighbor]
            uniq_label = np.unique(neighbor_label)
            modular_contrib = np.zeros(uniq_label.shape[0])

            if uniq_label.shape[0] < 1: continue
            
            for nth, l in enumerate(uniq_label):
                modular_contrib[nth] = (G[neighbor, n] - GP[neighbor, n]).dot(l == neighbor_label)
                
            #Glabel[n] = uniq_label[modular_contrib.argmax()]
            
            Glabel[n] = np.random.choice(uniq_label[modular_contrib == modular_contrib.max()], 1)[0]
        #print(modular_contrib, np.random.choice(uniq_label[modular_contrib == modular_contrib.max()]))
        modular_new = modularity(G, GP, Glabel, m)
        modular_gain = modular_new - modular_cur
        modular_cur = modular_new
        #print(modular_gain)
        loop += 1
        
    Vsimple = np.unique(Glabel)
    Esimple = np.zeros((Vsimple.shape[0], Vsimple.shape[0]))
    #Esimple = []
    
    for n in Gnode:
        neighbor = np.where(G[:, n] == 1)[0]
        
        for neig in neighbor:
            if Glabel[n] != Glabel[neig]:
                #Esimple.append((np.where(Vsimple == Glabel[n])[0], np.where(Vsimple == Glabel[neig])[0]))
                Esimple[Vsimple == Glabel[n], Vsimple == Glabel[neig]] = 1
                Esimple[Vsimple == Glabel[neig], Vsimple == Glabel[n]] = 1

    g = nx.Graph(Esimple)
        
    return Glabel, Vsimple, g.edges()
