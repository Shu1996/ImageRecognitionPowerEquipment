import tensorflow as tf
import data_process
from PIL import Image
import numpy as np
import cv2


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 10000])
y_ = tf.placeholder("float", shape=[None, 2])
keep_prob = tf.placeholder("float")

def CNNmodel():

#conv1
    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])

    x_image = tf.reshape(x, [-1,100,100,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

#conv2
    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

#fully connected
    W_fc1 = weight_variable([25 * 25 * 32, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv


# Train and Test
def train_cnn():

    sess = tf.InteractiveSession()
    """
    获取数据集
    """
    data=data_process.feed_data()
    testData=data.get_test_data()

    y_conv=CNNmodel()
    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    print("卷积核5*5，16；池化层2*2；batch size = 600; epoch = 50")
    print("下面是训练集准确度：")
    for i in range(2501):
        batch = data.get_train_data()
        if i%5 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print(train_accuracy)
            # print("step %d, training accuracy %g" % (i, train_accuracy))
            if i%100 == 0:
                print('step %d' %i)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    saver = tf.train.Saver()
    saver.save(sess,'model_16_600_50e/cnn3.ckpt')

    print("下面是测试集准确度(每20个一组)")
    for i in range(101):
        testData = data.get_test_data()
        if i%20 == 0:
            test_accuracy = accuracy.eval(feed_dict={x: testData[0], y_: testData[1], keep_prob: 1.0})
            print(test_accuracy)
            # print("step %d: test accuracy %g" % (i, test_accuracy))

def predict(data):
    y_conv = CNNmodel()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'model_16conv/cnn3.ckpt')  # restore model
    prediction = tf.argmax(y_conv, 1)
    result=prediction.eval(session=sess,feed_dict={x:data,keep_prob:1.0})
    print(result)

def eval():
    path = 'E:\\python\\project\\4.jpg'
    img = Image.open(path)
    img1 = img.convert('L')
    data = img1.getdata()
    data = np.matrix(data, dtype='float32')
    # data = np.array(data)
    predict(data)
    im = cv2.imread(path)
    cv2.namedWindow(path, 0)
    cv2.startWindowThread()
    cv2.imshow(path,im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

train_cnn()
# eval()





