import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

def add_layer(inputs, input_size, output_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([input_size, output_size]))
    threshold = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + threshold
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights

inputSize = 1
hiddenSize = 10
dataVolume = 300

x_data = 2 * np.random.random_sample((dataVolume, inputSize)) -1;
x_data = np.sort(x_data, 0)
noise = np.random.normal(0, 0.3, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
xs = tf.placeholder(tf.float32, [dataVolume, inputSize]);
ys = tf.placeholder(tf.float32);

hiddenLayer, weight_h = add_layer(xs, inputSize, hiddenSize, activation_function = tf.tanh)
outputLayer, weight_o = add_layer(hiddenLayer, hiddenSize, 1, activation_function = None)
error = tf.reduce_mean(tf.reduce_sum(tf.square(ys - outputLayer), reduction_indices=[1]))
learning_rate = tf.placeholder(tf.float32, shape=[])
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
lr=0.1
sess.run(train, feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
#out = sess.run([error, grad_w1, grad_w2], feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
tmp = sess.run([error, weight_h, weight_o], feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
temp_loss, temp_weight_h, temp_weight_o = tmp
#temp_loss, grad_w1_val, grad_w2_val = out
learning_times = []
#learning_times.append(1)
lAry = []
#sess.run(weight1.assign(weight1 - lr * grad_w1_val))
#sess.run(weight2.assign(weight2 - lr * grad_w2_val))
i = 1
while (i > 0):
    sess.run(train, feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
    #out = sess.run([error, grad_w1, grad_w2], feed_dict={xs: x_data, ys: y_data, learning_rate: lr})    
    #loss, grad_w1_val, grad_w2_val = out
    tmp = sess.run([error, weight_h, weight_o], feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
    loss, wh, wo = tmp
    if(loss < temp_loss):
        if(loss < 0.097):
            print("----------已達成學習目標----------")
            break
        lr = lr * 1.2
        temp_weight_h = wh
        temp_weight_o = wo
        temp_loss = loss
        lAry.append(loss)
        learning_times.append(i)
        i = i + 1
        print("loss: ", loss)
        
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(outputLayer, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', linewidth=5)
        plt.pause(0.1)
        
    else:
        if (lr > 0.05):
            lr = lr * 0.7
            sess.run(weight_h.assign(temp_weight_h))
            sess.run(weight_o.assign(temp_weight_o))
        else:
            print("----------Claim Undesired Attractor----------")
            break  
    #learning_times.append(1)
    #print("learning rate: ", lr)
#print("learning_times: ", learning_times)         
sess.close();

#abGraph = plt2.plot(learning_times, lAry)
#plt.show()