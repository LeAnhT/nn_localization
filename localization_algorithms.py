import numpy as np
import data_creator as dc
import measuring_tools as mt
import plot_helper as ph

import nn_trainer as nt


# implementierung der Quasi-Linearisierung
def TWR_2D(distances, anchors):
    anchor_matrix = np.array(anchors)
    x = anchor_matrix[:,0]
    y = anchor_matrix[:,1]
    k = np.add(np.square(x),np.square(y))
    #initialisiere A und b
    A = np.arange((len(anchors)-1)*len(anchors[0]),dtype=float)
    A = A.reshape((len(anchors)-1), len(anchors[0]))
    b = np.arange((len(anchors)-1),dtype=float)

    #Berechnungen
    for i in range(1, len(distances)):
        A[i-1,0] = x[i] - x[0]
        A[i-1,1] = y[i] - y[0]
        b[i - 1] = np.square(distances[0]) - np.square(distances[i]) - k[0] + k[i]
    solution_vect = (np.dot(np.linalg.pinv(A),np.transpose(b)))/2
    return solution_vect


def NN_2D(distances, anchors,metafile_path,checkpoint_path):
    import tensorflow as tf
    total_features = np.array(dc.prepare_input(anchors,distances))
    total_features = total_features.reshape(1,len(total_features))
    with tf.Session() as eval_sess:
        saver = tf.train.import_meta_graph(metafile_path)
        saver.restore(eval_sess, tf.train.latest_checkpoint(checkpoint_path))
        prediction = nt.make_prediction()
        solution_vect = eval_sess.run(prediction, feed_dict={nt.x: total_features})
    return solution_vect

fixed_anchors = [[0,0],[20,0],[0,20],[20,20]]
target = [10,10]
distance = mt.get_distance_to_anchors(target,fixed_anchors)
print(TWR_2D(distance,fixed_anchors))
print(NN_2D(distance,fixed_anchors,'C:/Anchors/2l_10cm_model/anchors_nn.meta','C:/Anchors/2l_10cm_model'))


'''
errors = []
for i in range(0,len(griddata)):
   # distances = mt.get_normalvariate_distance_to_anchors(griddata[i],fixed_anchors,0.10,0.01)
    distances = mt.get_distance_to_anchors(griddata[i],fixed_anchors)
    prediction = TWR_2D(distances,fixed_anchors)
    error = np.linalg.norm(np.array(prediction) - np.array(griddata[i]))
    errors.append(error)
    print(error)

ph.plot_ecdf(errors,'Error','Cumulative Percent')
ph.plot_grid_contour(20,20,0.1,errors,'X Position (cm)','Y Position (cm)')
ph.simple_plot(errors,'X th Prediction','Error')
'''




