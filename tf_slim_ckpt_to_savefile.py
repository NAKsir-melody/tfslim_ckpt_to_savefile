import tensorflow as tf
from tensorflow.contrib import slim
from nets import inception
from preprocessing import inception_preprocessing
from datasets import imagenet

import scipy
from scipy import misc
from numpy import array

image= scipy.misc.imread('woody.jpeg')
image= image.astype(float)
#scipy.misc.imshow(image)
image = array(image).reshape(1, 299,299,3)

input_image = tf.placeholder(tf.float32, shape=[None, 299,299,3])
img_scaled = tf.scalar_mul((1.0/255), input_image)
img_scaled = tf.subtract(img_scaled, 0.5)
img_scaled = tf.multiply(img_scaled, 2.0)

with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, endpoints = inception.inception_v3(img_scaled, num_classes=1001, is_training=False) 

    prediction = endpoints['Predictions']
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,'inception_v3.ckpt')
        pred_prob = sess.run(prediction, feed_dict = {input_image: image})


        saver.export_meta_graph('inception_v3.meta')
        saver.save(sess,'./inception_v3')

        # Getting the top 5 classes of the imagenet database
        probabilities = pred_prob[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        names = imagenet.create_readable_names_for_imagenet_labels()
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index]*100, names[index]))


