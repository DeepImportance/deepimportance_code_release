'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause

The purpose of this module is to demonstrate the process of obtaining pixel-wise explanations for given data points at hand of the MNIST hand written digit data set.

The module first loads a pre-trained neural network model and the MNIST test set with labels and transforms the data such that each pixel value is within the range of [-1 1].
The data is then randomly permuted and for the first 10 samples due to the permuted order, a prediction is computed by the network, which is then as a next step explained
by attributing relevance values to each of the input pixels.

finally, the resulting heatmap is rendered as an image and (over)written out to disk and displayed.
'''


import matplotlib.pyplot as plt
import numpy as np ; na = np.newaxis
import model_io
import data_io
import render
import sys, os
#sys.path.append('./../models/')
sys.path.append('./../')
from keras.layers import Input
from utils import load_MNIST


#load a neural network, as well as the MNIST test data and some labels

nn = model_io.read('../neural_networks/LeNet5.txt', 'txt') # 99.16% prediction accuracy
#nn = model_io.read('./models/MNIST/LeNet5.txt', 'txt') # 99.16% prediction accuracy
nn.drop_softmax_output_layer() #drop softnax output layer for analyses

X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)

x = X_test[50]

#forward pass and predictio
ypred = nn.forward(np.expand_dims(x, axis=0))
print(Y_test[50])
print(ypred)
print('Predicted Class:', np.argmax(ypred),'\n')

#prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
mask = np.zeros_like(ypred)
mask[:,np.argmax(ypred)] = 1
Rinit = ypred*mask
print(Rinit)

#compute first layer relevance according to prediction
#R, _ = nn.lrp(Rinit, 'simple')
#R, _ = nn.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
#R, _ = nn.lrp(Rinit,'epsilon',0.01)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
R, _ = nn.lrp(Rinit,'alphabeta',1)    #as Eq(60) from DOI: 10.1371/journal.pone.0130140

#R = R[:, 2:-2, 2:-2, :]

x = (x+1.)/2.

#render input and heatmap as rgb images
digit = render.digit_to_rgb(x, scaling = 3)
hm = render.hm_to_rgb(R, X = x, scaling = 3, sigma = 2) #
digit_hm = render.save_image([digit,hm],'../heatmap.png')
data_io.write(R,'../heatmap.npy')

#display the image as written to file
#plt.imshow(digit_hm, interpolation = 'none')
#plt.axis('off')
#plt.show()

