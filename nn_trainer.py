import numpy as np
import tensorflow as tf
import plot_helper as ph

#path-variables for saving and loading model
SAVE_FILE = 'C:/Anchors/2l_10cm_nv10cm_model/anchors_nn'
SAVE_DIRECTORY = 'C:/Anchors/2l_10cm_nv10cm_model'
LOAD_FILE = 'C:/Anchors/2l_10cm_nv10cm_model/anchors_nn.meta'


#optimization parameters
learning_rate = 0.00001
epochs = 100000

# grid-points as input, training-features = test-features
total_features = np.load('20x20_10cm_test_feats.npy')
total_labels = np.load('20x20_10cm_test_labels.npy')

# placeholders, "x" represents the input nodes(12 in total)
# "y" represents the output nodes (2 in total)
x = tf.placeholder(tf.float32,[None,12])
y = tf.placeholder(tf.float32,[None,2])

#weights between input-layer and hidden-layer1 (12x10 matrix)
w1 = tf.Variable(tf.truncated_normal([12,10],stddev=(2/ np.sqrt(12))))
#weights between hidden-layer1 and hidden-layer2 (10x10 matrix)
w2 = tf.Variable(tf.truncated_normal([10,10],stddev=(2/ np.sqrt(10))))
#weights between hidden-layer2 and outputlayer (10x2 matrix)
w3 = tf.Variable(tf.truncated_normal([10,2],stddev=(2/ np.sqrt(10))))

#bias for input layer
b1 = tf.Variable(tf.zeros(1))
#bias for hidden layer 1
b2 = tf.Variable(tf.zeros(1))
#bias for hidden layer 2
b3 = tf.Variable(tf.zeros(1))

def feed_forward():
    #result of the first hidden-layer
    hl1_out = tf.add(tf.matmul(x,w1),b1)
    #result of the second hidden-layer
    hl2_out = tf.add(tf.matmul(hl1_out,w2),b2)
    prediction = tf.add(tf.matmul(hl2_out,w3),b3)
    total_error = tf.reduce_mean(tf.square(y-prediction))
    return [prediction, total_error]

pred, total_err = feed_forward()

saver = tf.train.Saver()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_err)
init = tf.global_variables_initializer()



#---------- TRAIN NEURAL NET, run Tensorflow-Session, train the model and save the model under 'SAVE_FILE'
#---------- you might want to comment this session when evaluating the model
with tf.Session() as train_sess:
    train_sess.run(init)
    for i in range(epochs):
        train_sess.run(optimizer,feed_dict={x:total_features,y:total_labels})
    #save model
    train_sess.save(train_sess, SAVE_FILE)
train_sess.close()


#----------- TEST NEURAL NET, load trained model, feed it with the grid-data and output it's prediction

#use error_per_prediction for plotting the results e.g as ecdf, contour, etc...
error_per_prediction = []

#run tensorflow-session
with tf.Session() as eval_sess:
    #load trained model
    saver = tf.train.import_meta_graph(LOAD_FILE)
    saver.restore(eval_sess, tf.train.latest_checkpoint(SAVE_DIRECTORY))

    #we are just interested in the predictions, not the total_error, so we just run feed_forward()[0] and not feed_forward()
    pred = feed_forward()[0]
    prediction = eval_sess.run(pred, feed_dict={x:total_features, y:total_labels});

    #print the results and save the error per prediction for later plotting
    for i in range(0, len(total_features)):
        #error between i'th prediction and i'th label/true value
        error = np.linalg.norm(np.array(prediction[i]) - np.array(total_labels[i]))
        #print('prediction =', prediction[i])
        #print('true value =', total_labels[i])
        #print('error =', error)
        error_per_prediction.append(error)
eval_sess.close()

ph.plot_ecdf(error_per_prediction,'Error','Cumulative Percent')
ph.plot_grid_contour(20,20,0.1,error_per_prediction,'X Position (cm)','Y Position (cm)')
ph.simple_plot(error_per_prediction,'X th Prediction','Error')
