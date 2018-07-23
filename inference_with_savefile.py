#!/usr/bin/env python

import argparse
import numpy as np
from numpy import array
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import scipy
from scipy import misc


with tf.Graph().as_default() as graph: # Set default graph as graph
    with tf.Session() as sess:
                image = scipy.misc.imread('woody.jpeg')
                image = image.astype(float)
                #scipy.misc.imshow(image)
                image = array(image).reshape(1, 299,299,3)

                saver = tf.train.import_meta_graph('inception_v3.meta')

                l_output=graph.get_tensor_by_name("InceptionV3/Predictions/Softmax:0")
                l_input=graph.get_tensor_by_name("Placeholder:0")

                print "Shape of input : ", tf.shape(l_input)
                tf.train.Saver().restore(sess, 'inception_v3')
                pred_prob = sess.run( l_output, feed_dict = {l_input : image} )
                probabilities = pred_prob[0, 0:]

                with open("labels.txt") as f:
                    names = f.read().splitlines()

                sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

                for i in range(5):
                    index = sorted_inds[i]
                    print('Probability %0.2f%% => [%s]' % (probabilities[index]*100, names[index]))

