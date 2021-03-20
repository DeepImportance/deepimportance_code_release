import traceback
import os
import h5py
import sys
import datetime
import time
import random
import numpy as np
# import matplotlib.pyplot as plt
from cleverhans.attacks import SaliencyMapMethod, FastGradientMethod, CarliniWagnerL2, BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.datasets import mnist, cifar10
from keras.preprocessing import image
from keras.models import model_from_json
from keras.layers import Input
from keras.utils import np_utils
from keras import models
from lrp_toolbox.model_io import read
from neural_networks.dave_model import Dave_orig

random.seed(123)
np.random.seed(123)


def load_CIFAR(one_hot=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def load_MNIST(one_hot=True, channel_first=True):
    """
    Load MNIST data
    :param channel_first:
    :param one_hot:
    :return:
    """
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess dataset
    # Normalization and reshaping of input.
    if channel_first:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        # For output, it is important to change number to one-hot vector.
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def load_driving_data(path='driving_data/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'final_example.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    return train_xs, train_ys


def data_generator(xs, ys, target_size, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess_image(x, target_size)[0] for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess_image(x, target_size)[0] for x in paths]
            gen_state += batch_size
        yield np.array(X), np.array(y)


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


def deprocess_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_dave_model():
    # input image dimensions
    img_rows, img_cols = 100, 100
    input_shape = (img_rows, img_cols, 3)

    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)

    # load multiple models sharing same input tensor
    model = Dave_orig(input_tensor=input_tensor, load_weights=True)

    return model


def load_model(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into model
    model.load_weights(model_name + '.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Model structure loaded from ", model_name)
    return model


def get_layer_outs_old(model, class_specific_test_set):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    # Testing
    layer_outs = [func([class_specific_test_set, 1.]) for func in functors]

    return layer_outs


def get_layer_outs(model, test_input, skip=[]):
    inp = model.input  # input placeholder
    outputs = [layer.output for index, layer in enumerate(model.layers) \
               if index not in skip]

    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layer_outs = [func([test_input]) for func in functors]

    return layer_outs


def get_layer_outs_new(model, inputs, skip=[]):
    # TODO: FIX LATER. This is done for solving incompatibility in Simos' computer
    # It is a shortcut.
    # skip.append(0)
    evaluater = models.Model(inputs=model.input,
                             outputs=[layer.output for index, layer in enumerate(model.layers) \
                                      if index not in skip])

    # Insert some dummy value in the beginning to avoid messing with layer index
    # arrangements in the main flow
    # outs = evaluater.predict(inputs)
    # outs.insert(0, inputs)

    # return outs

    return evaluater.predict(inputs)


def calc_major_func_regions(model, train_inputs, skip=None):
    if skip is None:
        skip = []

    outs = get_layer_outs_new(model, train_inputs, skip=skip)

    major_regions = []

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        layer_out = layer_out.mean(axis=tuple(i for i in range(1, layer_out.ndim - 1)))

        major_regions.append((layer_out.min(axis=0), layer_out.max(axis=0)))

    return major_regions


def get_layer_outputs_by_layer_name(model, test_input, skip=None):
    if skip is None:
        skip = []

    inp = model.input  # input placeholder
    outputs = {layer.name: layer.output for index, layer in enumerate(model.layers)
               if (index not in skip and 'input' not in layer.name)}  # all layer outputs (except input for functionals)
    functors = {name: K.function([inp], [out]) for name, out in outputs.items()}  # evaluation functions

    layer_outs = {name: func([test_input]) for name, func in functors.items()}
    return layer_outs


def get_layer_inputs(model, test_input, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_input)

    inputs = []

    for i in range(len(outs)):
        weights, biases = model.layers[i].get_weights()

        inputs_for_layer = []

        for input_index in range(len(test_input)):
            inputs_for_layer.append(
                np.add(np.dot(outs[i - 1][0][input_index] if i > 0 else test_input[input_index], weights), biases))

        inputs.append(inputs_for_layer)

    return [inputs[i] for i in range(len(inputs)) if i not in skip]


def get_python_version():
    if (sys.version_info > (3, 0)):
        # Python 3 code in this block
        return 3
    else:
        # Python 2 code in this block
        return 2


# def show_image(vector):
#     img = vector
#     plt.imshow(img)
#     plt.show()


def save_quantization(qtized, filename, group_index):
    with h5py.File(filename + '_quantization.h5', 'w') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(qtized)):
            group.create_dataset("q" + str(i), data=qtized[i])

    print("Quantization results saved to %s" % (filename))
    return


def load_quantization(filename, group_index):
    try:
        with h5py.File(filename + '_quantization.h5', 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            qtized = []
            while True:
                # qtized.append(group.get('q' + str(i)).value)
                qtized.append(group.get('q' + str(i))[()])
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)
    except (AttributeError, TypeError) as error:
        print("Quantization results loaded from %s" % (filename))
        return qtized


def save_data(data, filename):
    with h5py.File(filename + '_dataset.h5', 'w') as hf:
        hf.create_dataset("dataset", data=data)

    print("Data saved to %s" % (filename))
    return


def load_data(filename):
    with h5py.File(filename + '_dataset.h5', 'r') as hf:
        dataset = hf["dataset"][:]

    print("Data loaded from %s" % (filename))
    return dataset


def save_layerwise_relevances(relevant_neurons, filename):
    with h5py.File(filename + '_relevant_neurons.h5', 'w') as hf:
        hf.create_dataset("relevant_neurons",
                          data=relevant_neurons)
    print("Relevant neurons saved to %s" % (filename))
    return


def load_layerwise_relevances(filename):
    with h5py.File(filename + '_relevant_neurons.h5',
                   'r') as hf:
        relevant_neurons = hf["relevant_neurons"][:]

    print("Layerwise relevances loaded from %s" % (filename))

    return relevant_neurons


def save_perturbed_test(x_perturbed, y_perturbed, filename):
    # save X
    with h5py.File(filename + '_perturbations_x.h5', 'w') as hf:
        hf.create_dataset("x_perturbed", data=x_perturbed)

    # save Y
    with h5py.File(filename + '_perturbations_y.h5', 'w') as hf:
        hf.create_dataset("y_perturbed", data=y_perturbed)

    print("Layerwise relevances saved to  %s" % (filename))
    return


def load_perturbed_test(filename):
    # read X
    with h5py.File(filename + '_perturbations_x.h5', 'r') as hf:
        x_perturbed = hf["x_perturbed"][:]

    # read Y
    with h5py.File(filename + '_perturbations_y.h5', 'r') as hf:
        y_perturbed = hf["y_perturbed"][:]

    return x_perturbed, y_perturbed


def save_perturbed_test_groups(x_perturbed, y_perturbed, filename, group_index):
    # save X
    filename = filename + '_perturbations.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("x_perturbed", data=x_perturbed)
        group.create_dataset("y_perturbed", data=y_perturbed)

    print("Classifications saved in ", filename)

    return


def load_perturbed_test_groups(filename, group_index):
    with h5py.File(filename + '_perturbations.h5', 'r') as hf:
        group = hf.get('group' + str(group_index))
        x_perturbed = group.get('x_perturbed').value
        y_perturbed = group.get('y_perturbed').value

        return x_perturbed, y_perturbed


def create_experiment_dir(experiment_path, model_name,
                          selected_class, step_size,
                          approach, susp_num, repeat):
    # define experiments name, create directory experiments directory if it
    # doesnt exist
    experiment_name = model_name + '_C' + str(selected_class) + '_SS' + \
                      str(step_size) + '_' + approach + '_SN' + str(susp_num) + '_R' + str(repeat)

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    return experiment_name


def save_classifications(correct_classifications, misclassifications, filename, group_index):
    filename = filename + '_classifications.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("correct_classifications", data=correct_classifications)
        group.create_dataset("misclassifications", data=misclassifications)

    print("Classifications saved in ", filename)
    return


def load_classifications(filename, group_index):
    filename = filename + '_classifications.h5'
    print
    filename
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            correct_classifications = group.get('correct_classifications').value
            misclassifications = group.get('misclassifications').value

            print("Classifications loaded from ", filename)
            return correct_classifications, misclassifications
    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)


def save_totalR(totalR, filename, group_index):
    filename = filename + '_relevances.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(totalR)):
            group.create_dataset("totalR_" + str(i), data=totalR[i])

    print("total relevance data saved in ", filename)
    return


def load_totalR(filename, group_index):
    filename = filename + '_relevances.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            totalR = []
            while True:
                # totalR.append(group.get('totalR_' + str(i)).value)
                totalR.append(group.get('totalR_' + str(i))[()])
                i += 1

    except (IOError) as error:
        print("File %s does not exist" % (filename))
        # print("Could not open file: ", filename)
        # traceback.print_exc()
        return None
    except (AttributeError, TypeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("totalR loaded from ", filename)
        return totalR


def save_layer_outs(layer_outs, filename, group_index):
    filename = filename + '_layer_outs.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(layer_outs)):
            group.create_dataset("layer_outs_" + str(i), data=layer_outs[i])

    print("Layer outs saved in ", filename)
    return


def load_layer_outs(filename, group_index):
    filename = filename + '_layer_outs.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            layer_outs = []
            while True:
                # layer_outs.append(group.get('layer_outs_' + str(i)).value)
                layer_outs.append(group.get('layer_outs_' + str(i))[()])
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except (AttributeError, TypeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("Layer outs loaded from ", filename)
        return layer_outs


def filter_correct_classifications(model, X, Y):
    X_corr = []
    Y_corr = []
    X_misc = []
    Y_misc = []
    preds = model.predict(X)  # np.expand_dims(x,axis=0))

    for idx, pred in enumerate(preds):
        if np.argmax(pred) == np.argmax(Y[idx]):
            X_corr.append(X[idx])
            Y_corr.append(Y[idx])
        else:
            X_misc.append(X[idx])
            Y_misc.append(Y[idx])

    '''
    for x, y in zip(X, Y):
        if np.argmax(p) == np.argmax(y):
            X_corr.append(x)
            Y_corr.append(y)
        else:
            X_misc.append(x)
            Y_misc.append(y)
    '''

    return np.array(X_corr), np.array(Y_corr), np.array(X_misc), np.array(Y_misc)


def filter_val_set(desired_class, X, Y):
    """
    Filter the given sets and return only those that match the desired_class value
    :param desired_class:
    :param X:
    :param Y:
    :return:
    """
    X_class = []
    Y_class = []
    for x, y in zip(X, Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)
    print("Validation set filtered for desired class: " + str(desired_class))
    return np.array(X_class), np.array(Y_class)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def get_trainable_layers(model):
    trainable_layers = []
    for idx, layer in enumerate(model.layers):
        try:
            if 'input' not in layer.name and 'softmax' not in layer.name and \
                    'pred' not in layer.name and 'drop' not in layer.name:
                weights = layer.get_weights()[0]
                trainable_layers.append(model.layers.index(layer))
        except:
            pass

    # trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return trainable_layers


def weight_analysis(model, target_layer):
    threshold_weight = 0.1
    deactivatables = []
    for i in range(2, target_layer + 1):
        for k in range(model.layers[i - 1].output_shape[1]):
            neuron_weights = model.layers[i].get_weights()[0][k]
            deactivate = True
            for j in range(len(neuron_weights)):
                if neuron_weights[j] > threshold_weight:
                    deactivate = False

            if deactivate:
                deactivatables.append((i, k))

    return deactivatables


def percent_str(part, whole):
    return "{0}%".format(float(part) / whole * 100)


def generate_adversarial(original_input, method, model,
                         target=None, target_class=None, sess=None, **kwargs):
    if not hasattr(generate_adversarial, "attack_types"):
        generate_adversarial.attack_types = {
            'fgsm': FastGradientMethod,
            'jsma': SaliencyMapMethod,
            'cw': CarliniWagnerL2,
            'bim': BasicIterativeMethod
        }

    if sess is None:
        sess = K.get_session()

    if method in generate_adversarial.attack_types:
        attacker = generate_adversarial.attack_types[method](KerasModelWrapper(model), sess)
    else:
        raise Exception("Method not supported")

    if type(original_input) is list:
        original_input = np.asarray(original_input)
    else:
        original_input = np.asarray([original_input])

        if target_class is not None:
            target_class = [target_class]

    if target is None and target_class is not None:
        target = np.zeros((len(target_class), model.output_shape[1]))
        target[np.arange(len(target_class)), target_class] = 1

    if target is not None:
        kwargs['y_target'] = target

    return attacker.generate_np(original_input, **kwargs)


def find_relevant_pixels(inputs, model_path, lrpmethod, relevance_percentile):
    lrpmodel = read(model_path + '.txt', 'txt')  # 99.16% prediction accuracy
    lrpmodel.drop_softmax_output_layer()  # drop softnax output layer for analysis

    all_relevant_pixels = []

    for inp in inputs:
        ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))

        mask = np.zeros_like(ypred)
        mask[:, np.argmax(ypred)] = 1
        Rinit = ypred * mask

        if lrpmethod == 'simple':
            R_inp, R_all = lrpmodel.lrp(Rinit)  # as Eq(56) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'epsilon':
            R_inp, R_all = lrpmodel.lrp(Rinit, 'epsilon', 0.01)  # as Eq(58) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'alphabeta':
            R_inp, R_all = lrpmodel.lrp(Rinit, 'alphabeta', 3)  # as Eq(60) from DOI: 10.1371/journal.pone.0130140

        if 'lenet' in model_path.lower():
            R_inp_flat = R_inp.reshape(28 * 28)
        elif 'cifar' in model_path.lower():
            R_inp_flat = R_inp.reshape(32 * 32 * 3)
        else:
            R_inp_flat = R_inp.reshape(100 * 100 * 3)

        abs_R_inp_flat = np.absolute(R_inp_flat)

        relevance_threshold = np.percentile(abs_R_inp_flat, relevance_percentile)
        # if relevance_threshold < 0: relevance_threshold = 0

        s = datetime.datetime.now()
        if 'lenet' in model_path.lower():
            R_inp = np.absolute(R_inp.reshape(28, 28))
        elif 'cifar' in model_path.lower():
            R_inp = np.absolute(R_inp.reshape(32, 32, 3))
        else:
            R_inp = np.absolute(R_inp.reshape(100, 100, 3))

        relevant_pixels = np.where(R_inp > relevance_threshold)
        all_relevant_pixels.append(relevant_pixels)
    return all_relevant_pixels


def save_relevant_pixels(filename, relevant_pixels):
    with h5py.File(filename + '_relevant_pixels.h5', 'a') as hf:
        group = hf.create_group('gg')
        for i in range(len(relevant_pixels)):
            group.create_dataset("relevant_pixels_" + str(i), data=relevant_pixels[i])

    print("Relevant pixels saved to %s" % (filename))
    return


def load_relevant_pixels(filename):
    try:
        with h5py.File(filename + '_relevant_pixels.h5', 'r') as hf:
            group = hf.get('gg')
            i = 0
            relevant_pixels = []
            while True:
                # relevant_pixels.append(group.get('relevant_pixels_' + str(i)).value)
                relevant_pixels.append(group.get('relevant_pixels_' + str(i))[()])
                i += 1
    except (AttributeError, TypeError) as error:
        # because we don't know the exact number of inputs in each class
        # we leave it to iterate until it throws an attribute error, and then return
        # return relevant pixels to the caller function

        print("Relevant pixels loaded from %s" % (filename))

        return relevant_pixels


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)