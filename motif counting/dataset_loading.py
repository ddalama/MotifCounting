import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data=loadmat(r"/Users/dheepdalamal/Downloads/BP.mat")
#print(data)
#load the mat files

#x = data["label"]
#print(x)
print(data["fmri"][0])
#print(type(x))

#print(data["fmri"].shape)

for key in data.keys():
    print(key)


## steps: parse mat file and convert to graph in networkx. implement edit distance graph. then see what to do (maybe ask).

#con_list = [[element for element in upperElement] for upperElement in x['obj_contour']]
#print(con_list)