from utils import get_layer_outs_new, preprocess_image, get_trainable_layers
from utils import load_driving_data, load_dave_model, load_totalR, load_relevant_pixels

from operator import itemgetter
import numpy as np

np.random.seed(1)

experiment_folder = "experiments"
# model_path = "neural_networks/LeNet5"
# model_path = "neural_networks/cifar_original"
model_path = "neural_networks/dave2"
# num_relevant_neurons = 6 #for others
num_relevant_neurons = 8  # for dave
# selected_class = 9 #lenet5
# selected_class = 1 #Lenet4
# selected_class = 9 #Lenet1
# selected_class = 4 #Cifar
selected_class = -1  # dave

model_name = model_path.split('/')[1]

'''
X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
img_rows, img_cols = 28, 28

X_train, Y_train, X_test, Y_test = load_CIFAR()
img_rows, img_cols = 32, 32

X_test, Y_test = filter_val_set(selected_class, X_test, Y_test)
'''

X_all = []
X_paths, Ys = load_driving_data()
for xp in X_paths:
    X_all.append(preprocess_image(xp)[0])

inputs = X_all[4000:]

'''
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
'''

model = load_dave_model()

subject_layer = get_trainable_layers(model)[-1]

fname = '%s/%s_%s_%d_%d' % (experiment_folder, model_name, 'alphabeta', selected_class, 98)

all_relevant_pixels = load_relevant_pixels(fname)

# TODO: Replace zip  with nested loops
minps = []
for idx, inp in enumerate(inputs):
    minp = inp.copy()
    relevant_pixels = all_relevant_pixels[idx]
    for i, j, k in zip(relevant_pixels[0], relevant_pixels[1], relevant_pixels[2]):
        if minp[i][j][k] > 127.5:
            minp[i][j][k] = 0
        else:
            minp[i][j][k] = 255
    minps.append(minp)

mouts = get_layer_outs_new(model, np.array(minps))
outs = get_layer_outs_new(model, np.array(inputs))

totalR = load_totalR('%s/%s_%s_%d' % (experiment_folder, model_name,
                                      'totalR', selected_class), 0)

relevant_neurons = np.argsort(totalR[subject_layer])[0][::-1][:num_relevant_neurons]
least_relevant_neurons = np.argsort(totalR[subject_layer])[0][:num_relevant_neurons]

# TODO: Subject layer should be selected automatically
subject_layer = -4

random_neurons = np.random.choice(list(set(range(outs[subject_layer].shape[-1])) - set(relevant_neurons)),
                                  num_relevant_neurons, replace=False)

change_rel = []
change_rand = []
for i in range(len(inputs)):
    before_rand = np.array(itemgetter(*random_neurons)(outs[subject_layer][i]))
    after_rand = np.array(itemgetter(*random_neurons)(mouts[subject_layer][i]))
    before_rel = np.array(itemgetter(*relevant_neurons)(outs[subject_layer][i]))
    after_rel = np.array(itemgetter(*relevant_neurons)(mouts[subject_layer][i]))

    change_rel.append(np.linalg.norm(before_rel - after_rel))
    change_rand.append(np.linalg.norm(before_rand - after_rand))

# TODO: Watchout filename
ff = open('rel_dave.csv', 'w')
for crel, crand in zip(change_rel, change_rand):
    ff.write(str(crel) + ',' + str(crand) + '\n')
