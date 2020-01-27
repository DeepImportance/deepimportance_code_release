import os
import sys

import datetime
import random
import argparse
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import model_from_json, load_model

from manipulate import add_wn_all, add_wn_frame, add_white_noise, add_wn_random
from manipulate import block_relevant_pixels, block_random_pixels
from utils import load_MNIST, load_CIFAR
from utils import filter_val_set, get_trainable_layers
from utils import save_layerwise_relevances, load_layerwise_relevances
from utils import save_layer_outs, load_layer_outs, get_layer_outs_new
from utils import save_data, load_data, save_quantization, load_quantization
from utils import generate_adversarial, filter_correct_classifications
from coverages.comb_cov import CombCoverage
from coverages.neuron_cov import NeuronCoverage
from coverages.tkn import DeepGaugeLayerLevelCoverage
from coverages.kmn import DeepGaugePercentCoverage
from coverages.ss import SSCover
from coverages.sa import SurpriseAdequacy
from neural_networks.small_mnist_model import SmallModel
from lrp_toolbox.model_io import write, read

experiment_folder = 'experiments'
dataset = 'mnist'

model_names   = ['neural_networks/LeNet1', 'neural_networks/LeNet4', 'neural_networks/LeNet5']# 'neural_networks/cifar40_128']
labels        = [0,1,2,3,4,5,6,7,8,9]
model_path = 'neural_networks/LeNet5'
selected_class = 5
adv_type = 'fgsm'
num_rel_neurons = 8
subject_layer = 7 #6 and 7 for l4 an l5

def lm(model_path):
    try:
        json_file = open(model_path + '.json', 'r') #Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        model = model_from_json(file_content)
        model.load_weights(model_path + '.h5')

        # Compile the model before using
        model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    except:
        model = load_model(model_path + '.h5')

    return model

for mp in model_paths:
    for adv in adv_types:
        ####################
        # 0) Load MNIST or CIFAR10 data
        if dataset == 'mnist':
            X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
            img_rows, img_cols = 28, 28
        else:
            X_train, Y_train, X_test, Y_test = load_CIFAR()
            img_rows, img_cols = 32, 32

        if not selected_class == -1:
            X_train_filtered, Y_train_filtered = filter_val_set(selected_class, X_train, Y_train) #Get training input for selected_class
            X_test_filtered, Y_test_filtered = filter_val_set(selected_class, X_test, Y_test) #Get testing input for selected_class


        ####################
        # 1) Setup the model
        model_name = model_path.split('/')[-1]

        model = lm(model_path)

        maninp1 = block_relevant_pixels(X_test_filtered, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=80)
        maninp2 = block_random_pixels(X_test_filtered, 156)

        evl = model.evaluate(X_test_filtered,Y_test_filtered)
        print(evl)
        evl = model.evaluate(np.array(maninp1[100:]),Y_test_filtered[100:])
        print(evl)

        model.fit(np.concatenate((maninp2[:100],X_train)), np.concatenate((Y_test_filtered[:100],Y_train)), verbose=False)

        evl = model.evaluate(np.array(maninp1[100:]),Y_test_filtered[100:])
        print(evl)

exit()

'''
X_adv_0 = load_data('%s/%s_%d_%s_adversarial' %(experiment_folder, model_name,
                                              selected_class, adv_type))

X_adv = load_data('%s/%s_%d_%s_adversarial' %(experiment_folder, model_name,
                                              -1, adv_type))

evl = model.evaluate(X_test,Y_test)
print(evl)

evl = model.evaluate(X_adv,Y_test)
print(evl)

evl = model.evaluate(X_adv_0,Y_test_filtered)
print(evl)
'''
#maninp = add_white_noise(X_test_filtered, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)
#maninp = add_wn_random(X_test_filtered, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)
#maninp2 = add_white_noise(X_test_filtered, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)
#maninp3 = add_white_noise(X_test_filtered, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)


#maninp = np.concatenate((maninp1,maninp2,maninp3))
Y_test_comp = Y_test_filtered# np.concatenate((Y_test_filtered, Y_test_filtered, Y_test_filtered))

model.fit(np.concatenate((maninp,X_train)), np.concatenate((Y_test_comp,Y_train)), verbose=False)

evl = model.evaluate(X_test,Y_test)
print(evl)

evl = model.evaluate(X_adv,Y_test)
print(evl)

evl = model.evaluate(X_adv_0,Y_test_filtered)
print(evl)

exit()

X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model,
                                                                   X_train,
                                                                   Y_train)

cc = CombCoverage(model, model_name, num_rel_neurons, selected_class,
                  subject_layer, X_train_corr, Y_train_corr)

orig_coverage, _, _ = cc.test(X_test[:len(X_test)/2])

cnt = 0
inc_adv = []
non_adv = []
yinc = []
ynon = []
for idx, xa in enumerate(X_adv):
    X_temp = np.concatenate((X_test[:len(X_test)/2], np.expand_dims(xa, axis=0)))
    adv_coverage, _, _ = cc.test(X_temp)
    if adv_coverage > orig_coverage:
        inc_adv.append(xa)
        yinc.append(Y_test[idx])
    else:
        non_adv.append(xa)
        ynon.append(Y_test[idx])

    if len(inc_adv) > 19 and len(non_adv) > 19: break

    print(len(inc_adv))

inc_num = len(inc_adv)

xx1 = np.concatenate((inc_adv,X_train))
xx2 = np.concatenate((non_adv[:inc_num],X_train))

yy1 = np.concatenate((yinc,Y_train))
yy2 = np.concatenate((ynon[:inc_num],Y_train))

model.fit(xx1,yy1,epochs=3, verbose=False)

evl = model.evaluate(X_adv, Y_test)
print(evl)

model = lm(model_path)

model.fit(xx2,yy2,epochs=3, verbose=False)

evl = model.evaluate(X_adv, Y_test)
print(evl)

exit()


'''
evl = model.evaluate(X_adv,Y_test)
print(evl)

maninp1 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)
maninp2 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)
maninp3 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)
maninp4 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)
maninp5 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98, noise_std_dev=0.75)
maninp = np.concatenate((maninp1,maninp2,maninp3,maninp4,maninp5))
many = np.concatenate((Y_test,Y_test,Y_test,Y_test,Y_test))

model.fit(np.array(maninp), many, epochs=100, verbose=False)

evl = model.evaluate(X_adv,Y_test)
print(evl)
'''

