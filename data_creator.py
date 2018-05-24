#create training-data

import numpy as np
import measuring_tools as mt

# create targetgrid of given x,y-length and step
def create_targetgrid(x_width,y_width,step):
    #x_width & y_width + step, um x_width/y_width mit einzuschließen
    #(arange würde x_width/y_width ausschließen)
    X = np.arange(0, x_width+step, step)
    Y = np.arange(0, y_width+step, step)
    grid = []
    for i in range(0,len(Y)):
        for j in range(0,len(X)):
            grid.append((X[j],Y[i]))
    return grid

#prepare NN-train-dataset by combining anchor-and distance data
def prepare_input(anchors, distances):
    feature = []
    for i in range(len(anchors)):
        feature.append(anchors[i][0])
        feature.append(anchors[i][1])
    for i in range(len(distances)):
        feature.append(distances[i])
    return feature

#create training-dataset with fixed numbers of anchors (4)
def generate_fixedgrid_traindata(x, y,steps):
    #fixed_anchors = [[5,5],[15,5],[5,15],[15,15]]
    fixed_anchors = [[0,0],[x,0],[0,y],[x,y]]
    feats = []
    labels = []
    grid_elements = create_targetgrid(x,y,steps)
    for i in range(len(grid_elements)):
        #distances = mt.get_distance_to_anchors(grid_elements[i], fixed_anchors)
        distance_with_error = mt.get_normalvariate_distance_to_anchors(grid_elements[i],fixed_anchors,0.10,0.01)
        feats.append(prepare_input(fixed_anchors,distance_with_error))
        labels.append(grid_elements[i])
    return [feats,labels]

#feat, res = generate_fixedgrid_traindata(20,20,0.1)
#np.save('20x20_10cm_test_feats',feat)
#np.save('20x20_10cm_test_labels',res)

