import numpy as np
from keras.models import model_from_json

from utils import load_MNIST
from utils import filter_val_set, get_trainable_layers
from utils import filter_correct_classifications
from coverages.idc import CombCoverage
from reproduce.manipulate import add_white_noise

X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
img_rows, img_cols = 28, 28

####################
##### LENET-1
#####################

model_path = 'neural_networks/LeNet1'
model_name = 'LeNet1'
num_rel_neurons = 4

####################
##### LENET-4
#####################

#model_path = 'neural_networks/LeNet4'
#model_name = 'LeNet4'
#num_rel_neurons = 4

####################
##### LENET-5
#####################

model_path = 'neural_networks/LeNet5'
model_name = 'LeNet5'
num_rel_neurons = 6


json_file = open(model_path + '.json', 'r') #Read Keras model parameters (stored in JSON file)
file_content = json_file.read()
json_file.close()

model = model_from_json(file_content)
model.load_weights(model_path + '.h5')

# Compile the model before using
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


trainable_layers = get_trainable_layers(model)
non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
print('Trainable layers: ' + str(trainable_layers))
print('Non trainable layers: ' + str(non_trainable_layers))


experiment_folder = 'experiments'


X_train_corr, Y_train_corr, _, _ = filter_correct_classifications(model,
                                                                   X_train,
                                                                   Y_train)

fw = open('layer_sens.log', 'a')

for subject_layer in range(len(trainable_layers)):
    subject_layer = trainable_layers[subject_layer]
    for selected_class in [0,1,2,3,4]:#,5,6,7,8,9]:

        X_train_class, Y_train_class = filter_val_set(selected_class, X_train_corr, Y_train_corr)
        X_test_class, Y_test_class   = filter_val_set(selected_class, X_test, Y_test)
        X_test_corr,  Y_test_corr, _, _  = filter_correct_classifications(model, X_test_class, Y_test_class)

        cc = CombCoverage(model, model_name, num_rel_neurons, selected_class,
                        subject_layer, X_train_class, Y_train_class)

        orig_coverage, orig_covered_combinations = cc.test(X_test_class)

        cc.set_measure_state(orig_covered_combinations)

        maninp  = add_white_noise(X_test_corr, model_path, selected_class, lrpmethod='alphabeta') #CHANGE TO 0.4 FOR MNIST

        rel_coverage, rel_covered_combinations = cc.test(np.array(maninp))

        res = {
            'model_name': model_name,
            'layer': subject_layer,
            'class': selected_class,
            # 'qgran': quantization_granularity,
            'orig_coverage': orig_coverage,
            'orig_numcov': len(orig_covered_combinations),
            'rel_coverage': rel_coverage,
            'rel_numcov': len(rel_covered_combinations),
        }

        fw.write(str(res))
        fw.write('\n')

fw.close()
