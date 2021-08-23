import networkx as nx
import numpy as np
import itertools
import nltk
from nltk.tokenize import WhitespaceTokenizer
import matplotlib.pyplot as plt
from scipy.io import loadmat
from time import time

from collections import Counter


"""
.By 7/27:       

TODO:
Use fmri (82,82,97), the fourth column in the node file to define motifs.
The fourth column provides node labels(functional module)
The edges +- (edge weight>threshold, positive, <-threshold,negative.  threshold= 0, 0.2, 0.5)
The edge weight between roi1 and roi2 is sample[0,1] or sample[1,0] np.fill_diagnose()
Check significance between classes defined in label.
2 nodes motif (write these motifs within significant differences among groups on readme.)
3 nodes motif 

Plan: 

Remake dataset by including 4th column. done
Enumerate motifs (do only 2 nodes) 
Find counts
Use t-tests
Publish results on readme


"""

"""
TODO by 8/9:

Find differences between classes and conduct t-test. Create table
 similar to in Dheep-Motif doc. 
 
Plan:
1) Test sample graph on motif counting alg. 
2) Input graphs to alg. and get each class's average count for each motif.
3) Create table. 
"""

## We define each S* motif as a directed graph in networkx


brain_motifs = {

}


# enumerating all possible 2-node brain motifs

    #all possible unsigned, unlabelled graphs
M1, M2 = nx.Graph(), nx.Graph()
M1.add_node(0)
M1.add_node(1)
M2.add_node(0)
M2.add_node(1)
M2.add_edge(0, 1)

unsigned_unlabelled_twonode_motifs = {'S1': M1, 'S2': M2}

nx.set_edge_attributes(M1, 'no sign added', 'sign')
nx.set_node_attributes(M1, -1, 'module')
nx.set_edge_attributes(M2, 'no sign added', 'sign')
nx.set_node_attributes(M2, -1, 'module')



    # all possible signed, unlabelled motifs
signed_motifs = set()
temp_motif = nx.Graph()
signed_motifs.add(M1.copy())
for motif in [M2]:
    for sign in ['positive', 'negative']:
        temp_motif = motif
        nx.set_edge_attributes(temp_motif, sign, 'sign')
        signed_motifs.add(temp_motif.copy())

    #all possible signed, labelled motifs
for motif in signed_motifs:
    for module in [1, 2, 3, 4, 5, 6]:
        temp_motif = motif.copy()


#M2 =
# temp_motif = nx.Graph()
# for graph in [M1, M2]:
#     for sign in ['positive', 'negative']:
#         temp_motif = M1
#         temp_motif[0]['sign'] = sign
#         signed_motifs.append(temp_motif)
print('')


motifs = {
    'S1': nx.DiGraph([(1, 2), (2, 3)]), #are these edges?
    'S2': nx.DiGraph([(1, 2), (1, 3), (2, 3)]),
    'S3': nx.DiGraph([(1, 2), (2, 3), (3, 1)]),
    'S4': nx.DiGraph([(1, 2), (3, 2)]),
    'S5': nx.DiGraph([(1, 2), (1, 3)])
}

size_three_brain_motifs = {
    'S1': nx.DiGraph([(0, 1), (1, 2), (1, 0), (2, 1)]),
    'S2': nx.DiGraph([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
}

# adjacency matrix for disconnected 2-node unsigned unlabelled motif.
a = [[0, 0], [0, 0]]
adj = np.array(a)

size_two_motifs = {
    'S1': nx.DiGraph([(0, 1), (1, 0)]),
    'S2': nx.DiGraph(adj)
}

#nx.draw(size_three_brain_motifs['S1'])
nx.draw(size_three_brain_motifs['S2'])

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

    #!!!!! could generalise the algorithm for k-size motifs
    tb = time()

    x = 0
    mcount = dict(zip(mo.keys(), list(map(int, np.zeros(len(mo))))))
    nodes = gr.nodes # what is this? generates NodeView obj of graph  #run


    # We use iterools.product to have all combinations of three nodes in the
    # original graph. Then we filter combinations with non-unique nodes, because
    # the motifs do not account for self-consumption.

    # What does sentence 2 mean?

    triplets = list(itertools.product(*[nodes, nodes, nodes])) #returns all 3-node combinations
    triplets = list([trip for trip in triplets if len(list(set(trip))) == 3]) #reduces list to include only 3-node combinations w no dups.
    triplets = map(list, map(np.sort, triplets))  # returns a list of sorted triplets (list of 3 nodes). maps isomorphic triplets to a single triplet. #run
    triplets = list(tuple(trip) for trip in triplets)
    u_triplets = list()
    u_triplets = list(set(triplets))

    # count = Counter()
    # for trip in triplets:
    #         count[trip] += 1
    #removes duplicate triplets. count is very slow. use counter object instead. OR could we use set() instead to elim dups.
    #    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]
    #From collections import counter

    # doubles = list(itertools.product(*[nodes, nodes]))
    # doubles = [doub for doub in doubles if len(list(set(doub))) == 2]
    # doubles = map(list, map(np.sort, doubles))
    # u_doubles = []
    # [u_doubles.append(doub) for doub in doubles if not u_doubles.count(doub)].
    # # The for each each of the triplets, we (i) take its subgraph, and compare
    # # it to all fo the possible motifs


    for trip in u_triplets:
        #print()
        t3 = time()
        t1 = time()
        sub_gr = gr.subgraph(trip)
        #nx.draw(sub_gr)
        t2 = time()
        td = t2 - t1

        # nx.draw(sub_gr)

        t1 = time()
        mot_match = list(map(lambda mot_id: nx.is_isomorphic(sub_gr, mo[mot_id]), mo.keys()))
        t2 = time()
        td = t2 - t1

        t1 = time()
        match_keys = [list(mo.keys())[i] for i in range(len(mot_match)) if mot_match[i]]
        t2 = time()
        td = t2 - t1

        t1 = time()
        if len(match_keys) == 1:
            mcount[match_keys[0]] += 1
        t2 = time()
        td = t2 - t1
        t4 = time()
        tf = t4 - t3
        #print("tf = " + str(tf))
        te = time()
        algtime = te - tb
        print("algtime = " + str(algtime))
    return mcount



def main():
    #copying 4th col of Brodmann txt file to a list called 'modules'
    tk = WhitespaceTokenizer()
    file = open('/Users/dheepdalamal/Downloads/Node_Brodmann82.txt', 'r')
    lines = []
    lines = file.readlines()
    modules = []
    tokenized_line = []
    for line in lines:
        tokenized_line = tk.tokenize(line)
        modules.append(tokenized_line[3])



    data = loadmat(r"/Users/dheepdalamal/Downloads/BP.mat")
    data_with_modules = data
    data_with_modules['module number'] = modules
    print(data_with_modules.keys())
    testGraph = data["fmri"][:, :, 0]






    DI = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])


    d = nx.DiGraph(testGraph)
    result = {}
    ti = time()
    print(ti)
    for matrix in data_with_modules["fmri"]:
        t1 = time()
        for row in matrix:
            for cell in row:
                if cell > 0.5 or cell < -0.5:
                    cell = 1
                else:
                    cell = 0
        t2 = time()
        t3 = t2 - t1
        print()

    tf = time()
    print(tf)

    td = (tf - ti)/60
    print(td)


    t1= time()
    print(t1)
    result = mcounter(d, size_three_brain_motifs)
    print(result)
    t2 = time()
    print(t2)
    tf = (t2 - t1)/60
    print(tf)



if __name__ == '__main__':
    main()