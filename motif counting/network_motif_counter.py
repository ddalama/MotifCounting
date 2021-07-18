import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.io import loadmat


## We define each S* motif as a directed graph in networkx


motifs = {
    'S1': nx.DiGraph([(1, 2), (2, 3)]), #are these edges?
    'S2': nx.DiGraph([(1, 2), (1, 3), (2, 3)]),
    'S3': nx.DiGraph([(1, 2), (2, 3), (3, 1)]),
    'S4': nx.DiGraph([(1, 2), (3, 2)]),
    'S5': nx.DiGraph([(1, 2), (1, 3)])
}


def mcounter(gr, mo): #does mo only contain motifs of size = 3?
    """Counts motifs in a directed graph
    :param gr: A ``DiGraph`` object
    :param mo: A ``dict`` of motifs to count
    :returns: A ``dict`` with the number of each motifs, with the same keys as ``mo``
    This function is actually rather simple. It will extract all 3-grams from
    the original graph, and look for isomorphisms in the motifs contained
    in a dictionary. The returned object is a ``dict`` with the number of
    times each motif was found.::
        >>> print mcounter(DiGraph gr, mo)
        {'S1': 4, 'S3': 0, 'S2': 1, 'S5': 0, 'S4': 3}
    """
    # This function will take each possible subgraphs of gr of size 3, then
    # compare them to the mo dict using .subgraph() and is_isomorphic

    # This line simply creates a dictionary with 0 for all values, and the
    # motif names as keys

    mcount = dict(zip(mo.keys(), list(map(int, np.zeros(len(mo))))))
    nodes = gr.nodes # what is this? generates NodeView obj of graph  #run


    # We use iterools.product to have all combinations of three nodes in the
    # original graph. Then we filter combinations with non-unique nodes, because
    # the motifs do not account for self-consumption.

    # What does sentence 2 mean?

    triplets = list(itertools.product(*[nodes, nodes, nodes])) #returns all 3-node combinations
    triplets = [trip for trip in triplets if len(list(set(trip))) == 3] #reduces list to include only 3-node combinations w no dups.
    triplets = map(list, map(np.sort, triplets))  # returns a list of sorted triplets (list of 3 nodes). maps isomorphic triplets to a single triplet. #run
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)] #removes duplicate triplets.

    # The for each each of the triplets, we (i) take its subgraph, and compare
    # it to all fo the possible motifs

#finished until here
    for trip in u_triplets:
        sub_gr = gr.subgraph(trip)
        mot_match = map(lambda mot_id: nx.is_isomorphic(sub_gr, mo[mot_id]), motifs.keys())
        match_keys = [mo.keys()[i] for i in range(len(mo)) if mot_match[i]]
        if len(match_keys) == 1:
            mcount[match_keys[0]] += 1

    return mcount

def test():
    listOfLists = [[1, 2, 3], [2, 1, 3], [3, 1, 2], [2, 3, 4], [3, 2, 4], [4, 2, 3]]
    shortenedList = []
    [shortenedList.append(trip) for trip in listOfLists if not shortenedList.count(trip)]
    print(shortenedList)


def main():
    #test()
    #G = nx.DiGraph([(1, 2), (2, 3)])
    #G.add_node(node_for_adding = 2)
    #G.add_node(3)
    #print(G.edges([1]))
    #print(G.nodes)
    #print("s")
    #H = G.subgraph([4, 5])
    #print(H.edges)
    #print("e")

    data = loadmat(r"/Users/dheepdalamal/Downloads/BP.mat")
    testGraph = data["fmri"][0]
    testGraphFinal = []
    for array in testGraph:
        testGraphFinal.append(list(array))

    diGraph = nx.DiGraph(testGraphFinal)
    print(diGraph.edges)
    print(mcounter(diGraph, motifs))

if __name__ == '__main__':
    main()


#loading dataset




