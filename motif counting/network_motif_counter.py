import networkx as nx
import numpy as np
import itertools
import nltk
import scipy
from nltk.tokenize import WhitespaceTokenizer
import matplotlib.pyplot as plt
from scipy.io import loadmat
from time import time
from scipy import stats

import collections

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


size_three_brain_motifs = {
    #3-node motifs
    'm3.1': nx.DiGraph([(0, 1), (1, 2), (1, 0), (2, 1)]),
    'm3.2': nx.DiGraph([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]),
    #4-node motifs

}

size_four_brain_motifs = {
    'm4.1': nx.DiGraph([(0, 1), (1, 2), (1, 0), (2, 3), (2, 1), (3, 2)]),
    'm4.2': nx.DiGraph([(0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (3, 1)]),
    'm4.3': nx.DiGraph([(0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 2), (3, 2)]),
    'm4.4': nx.DiGraph([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2)]),
    'm4.5': nx.DiGraph([(0, 1), (0, 3), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]),
    'm4.6': nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)])
}

# for key in size_three_brain_motifs:
#     nx.draw(size_three_brain_motifs[key])
#
# nx.draw(size_three_brain_motifs['m4.6'])



# adjacency matrix for disconnected 2-node unsigned unlabelled motif.
a = [[0, 0], [0, 0]]
adj = np.array(a)

size_two_motifs = {
    'S1': nx.DiGraph([(0, 1), (1, 0)]),
    'S2': nx.DiGraph(adj)
}



def variance(some_list):
    # Number of observations
    n = len(some_list)
    # Mean of the data
    mean = sum(some_list) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in some_list]
    # Variance
    variance = sum(deviations) / n

    return variance



def mcounter(gr, mo, size): #does mo only contain motifs of size = 3?
    """Counts motifs in a directed graph
    :param gr: A ``DiGraph`` object
    :param mo: A ``dict`` of motifs to count
    :param size: motif size
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

    node_repeat = [nodes] * size

    triplets = list(itertools.product(*node_repeat)) #returns all 3-node combinations
    triplets = list([trip for trip in triplets if len(list(set(trip))) == size]) #reduces list to include only 3-node combinations w no dups.
    triplets = map(list, map(np.sort, triplets))  # returns a list of sorted triplets (list of 3 nodes). maps isomorphic triplets to a single triplet. #run
    triplets = list(tuple(trip) for trip in triplets)
    u_triplets = list()
    u_triplets = list(set(triplets))

    for trip in u_triplets:

        sub_gr = gr.subgraph(trip)





        mot_match = list(map(lambda mot_id: nx.is_isomorphic(sub_gr, mo[mot_id]), mo.keys()))



        match_keys = [list(mo.keys())[i] for i in range(len(mot_match)) if mot_match[i]]



        if len(match_keys) == 1:
            mcount[match_keys[0]] += 1

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





    # node_repeat = list(node_repeat[:-2] + "]")
    print()
    print()
    data = loadmat(r"/Users/dheepdalamal/Downloads/BP.mat")
    data_with_modules = data
    data_with_modules['module number'] = modules

    mcounter(nx.DiGraph(data_with_modules["fmri"][:, :, 0]), size_three_brain_motifs, 4)




#making adjacency matrix binary
    for i in range(0, 97):
        t1 = time()
        for j in range(0, 82):
            for k in range(0, 82):
               if (data_with_modules["fmri"][:, :, i][j][k] > 0.005) or (data_with_modules["fmri"][:, :, i][j][k] < -0.005):
                    data_with_modules["fmri"][:, :, i][j][k] = 1
               else:
                    data_with_modules["fmri"][:, :, i][j][k] = 0
            data_with_modules["fmri"][:, :, i][j][j] = 0


    diseased_samples = 52
    non_diseased_samples = 45


    # #running mcounter alg. on whole dataset
    # diseased_aggregate = collections.Counter({'S1': 0, 'S2': 0})
    # non_diseased_aggregate = collections.Counter({'S1': 0, 'S2': 0})
    # diseased_average = collections.Counter({'S1': 0, 'S2': 0})
    # non_diseased_average = collections.Counter({'S1': 0, 'S2': 0})
    # diseased_list_S1 = []
    # non_diseased_list_S1 = []
    # diseased_list_S2 = []
    # non_diseased_list_S2 = []
    # for i in range(0, 97):
    #     if data_with_modules["label"][i] == 1:
    #         count = mcounter(nx.DiGraph(data_with_modules["fmri"][:, :, i]), size_three_brain_motifs)
    #         diseased_list_S1.append(count["S1"])
    #         diseased_list_S2.append(count["S2"])
    #         diseased_aggregate.update(count)
    #         print("diseased aggregate = " + str(diseased_aggregate))
    #
    #
    #     elif data_with_modules["label"][i] == -1:
    #         count = mcounter(nx.DiGraph(data_with_modules["fmri"][:, :, i]), size_three_brain_motifs)
    #         non_diseased_list_S1.append(count["S1"])
    #         non_diseased_list_S2.append(count["S2"])
    #         non_diseased_aggregate.update(count)
    #         print("non_diseased_aggregate = " + str(non_diseased_aggregate))
    #
    #
    # print()
    # print("______________________________________")
    # print()
    # print("diseased list S1 = " + str(diseased_list_S1))
    # print("diseased list S2 = " + str(diseased_list_S2))
    # print("non-diseased list S1 = " + str(non_diseased_list_S1))
    # print("non-diseased list S2 = " + str(non_diseased_list_S2))
    # diseased_average_S1 = diseased_aggregate["S1"]/diseased_samples
    # diseased_average_S2 = diseased_aggregate["S2"]/non_diseased_samples
    # non_diseased_average_S1 = non_diseased_aggregate["S1"]/non_diseased_samples
    # non_diseased_average_S2 = non_diseased_aggregate["S2"]/non_diseased_samples
    #
    # print("diseased average S1 = " + str(diseased_average_S1))
    # print("diseased average S2 = " + str(diseased_average_S2))
    # print("non diseased average S1 = " + str(non_diseased_average_S1))
    # print("non diseased average S2 = " + str(non_diseased_average_S2))
    #
    # print("diseased variance S1 = " + str(variance(diseased_list_S1)))
    # print("diseased variance S2 = " + str(variance(diseased_list_S2)))
    # print("non-diseased variance S1 = " + str(variance(non_diseased_list_S1)))
    # print("non-diseased variance S2 = " + str(variance(non_diseased_list_S2)))
    #
    #
    #
    # diseased_list_S1 = [1158, 1630, 1425, 1482, 1313, 1750, 1610, 1257, 1703, 1062, 1062, 1267, 1362, 1073, 1227, 1331, 1410, 972, 1225, 1545, 1200, 1221, 1362, 1074, 1426, 921, 1483, 1197, 1507, 1589, 1447, 1145, 1181, 1435, 1535, 1526, 1237, 1119, 1498, 1363, 1025, 1385, 1044, 1656, 1465, 1233, 1168, 1432, 1133, 1338, 1249, 1057]
    # diseased_list_S2 = [275, 365, 339, 302, 294, 461, 440, 240, 386, 280, 228, 291, 304, 269, 286, 295, 306, 238, 277, 372, 300, 319, 398, 305, 406, 215, 397, 299, 414, 376, 400, 305, 296, 381, 379, 378, 309, 277, 370, 299, 192, 317, 274, 402, 303, 314, 269, 326, 226, 372, 323, 249]
    # non_diseased_list_S1 = [1243, 1462, 1669, 1479, 1298, 1540, 1278, 1534, 1572, 1567, 1262, 1575, 1502, 1490, 1894, 1665, 1583, 1337, 1077,
    #  1959, 1500, 1386, 1804, 1265, 1962, 1423, 1232, 1068, 1346, 1246, 993, 1509, 1433, 1051, 1016, 1321, 1362, 1513,
    #  1739, 1338, 1034, 1239, 1379, 1039, 1038]
    # non_diseased_list_S2 = [263, 364, 435, 326, 313, 355, 307, 405, 398, 385, 276, 383, 363, 351, 468, 426, 396, 254, 264, 488, 308, 330, 442, 287, 521, 291, 303, 259, 313, 295, 192, 330, 338, 232, 256, 322, 302, 361, 436, 280, 248, 266, 341, 227, 245]
    #
    # S1_t_test = scipy.stats.ttest_ind(a = diseased_list_S1, b = non_diseased_list_S1, equal_var = False)
    # S2_t_test = scipy.stats.ttest_ind(a = diseased_list_S2, b = non_diseased_list_S2, equal_var = False)
    # print("S1 t-test = " + str(S1_t_test))
    # print("S2 t-test = " + str(S2_t_test))



if __name__ == '__main__':
    main()