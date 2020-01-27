import os
import sys

import numpy as np

from keras.models import model_from_json, load_model

import manipulate
import manipulate_cifar
from utils import load_MNIST, load_CIFAR
from utils import filter_val_set, get_trainable_layers
from utils import save_layerwise_relevances, load_layerwise_relevances
from utils import save_layer_outs, load_layer_outs, get_layer_outs_new
from utils import save_data, load_data, save_quantization, load_quantization
from utils import generate_adversarial, filter_correct_classifications


experiment_folder = 'experiments'
model_paths = ['neural_networks/LeNet1', 'neural_networks/LeNet4', 'neural_networks/LeNet5', 'neural_networks/cifar_original']
labels      = [0,1,2,3,4,5,6,7,8,9]

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

ff = open('robustness.log', 'w')

for mp in model_paths:
    if 'cifar' in mp: dataset = 'cifar10'
    else: dataset = 'mnist'

    for selected_class in labels:
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
        model_name = mp.split('/')[-1]

        model = lm(mp)

        orig_acc = model.evaluate(X_test_filtered,Y_test_filtered)[1]

        if dataset == 'mnist':
            maninp1 = manipulate.block_relevant_pixels(X_test_filtered, mp, selected_class, lrpmethod='alphabeta', relevance_percentile=95)
            maninp2 = manipulate.block_random_pixels(X_test_filtered, 40)
        else:
            maninp1 = manipulate_cifar.block_relevant_pixels(X_test_filtered, mp, selected_class, lrpmethod='alphabeta', relevance_percentile=96)
            maninp2 = manipulate_cifar.block_random_pixels(X_test_filtered, mp, selected_class)

        pert_acc = model.evaluate(np.array(maninp1[100:]),Y_test_filtered[100:])[1]

        model.fit(np.concatenate((maninp1[:100],X_train)), np.concatenate((Y_test_filtered[:100],Y_train)), verbose=False)

        rel_acc = model.evaluate(np.array(maninp1[100:]),Y_test_filtered[100:])[1]

        model = lm(mp)

        model.fit(np.concatenate((maninp2[:100],X_train)), np.concatenate((Y_test_filtered[:100],Y_train)), verbose=False)

        rand_acc = model.evaluate(np.array(maninp1[100:]),Y_test_filtered[100:])[1]

        res = {
            'orig_acc': orig_acc,
            'pert_acc': pert_acc,
            'rel_acc': rel_acc,
            'rand_acc': rand_acc,
            'model_name': model_name,
            'class': selected_class
        }

        ff.write(str(res))
        ff.write('\n')
ff.close()

