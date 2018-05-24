import numpy as np
from random import normalvariate
from random import gammavariate
import random
import time

#seed random
t = int( time.time() * 1000.0 )
random.seed( ((t & 0xff000000) >> 24) +
             ((t & 0x00ff0000) >>  8) +
             ((t & 0x0000ff00) <<  8) +
             ((t & 0x000000ff) << 24)   )

# return distance between 2 points
def get_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))
    #return math.sqrt( ((point1[0]-point2[0])**2)+((point1[1]-point2[1])**2) )

# returns list of distances between a targetpoint and a list of anchor-points
def get_distance_to_anchors(target_point, anchors):
    distances = []
    for point in anchors:
        distances.append(get_distance(point,target_point))
    return distances

# returns list of distances between a target-point and a list of anchor-points
# with normalvariate distributed error
def get_normalvariate_distance_to_anchors(target_point, anchors,sigma,mu):
    distances = []
    for point in anchors:
        distances.append(get_distance(point,target_point))
    for i in range(0,len(distances)):
        value = normalvariate(sigma,mu)
        #print("without error:" + str(distances[i]))
        #print("with error:" + str(distances[i]+value))
        distances[i]+=value
    return distances

# returns list of distances between a target-point and a list of anchor-points
# with gammavariate distributed error
def get_gammavariate_distance_to_refs(target_point,anchors,p,b):
    distances = []
    for point in anchors:
        distances.append(get_distance(point, target_point))
    for i in range(0, len(distances)):
        value = gammavariate(p,b)
        print("ohne aufschlag:" + str(distances[i]))
        distances[i] += value
    return distances

