from REBUTTAL_SCRIPTS.manipulate import add_wn_frame, add_white_noise, add_wn_random
from utils import load_MNIST, load_CIFAR
from utils import filter_val_set, get_trainable_layers
from utils import save_data, load_data
from utils import generate_adversarial, filter_correct_classifications
from coverages.comb_cov import CombCoverage
from coverages.neuron_cov import NeuronCoverage
from coverages.tkn import DeepGaugeLayerLevelCoverage
from coverages.kmn import DeepGaugePercentCoverage
from coverages.ss import SSCover
from coverages.sa import SurpriseAdequacy

import os
import random
import argparse
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import model_from_json, load_model

__version__ = 0.9


def batch_manipulate_rel(X_test, model_path, selected_class):
    maninp1 = np.array(
        add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                        noise_std_dev=0.75))
    maninp2 = np.array(
        add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                        noise_std_dev=0.75))
    print('yes')
    maninp3 = np.array(
        add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                        noise_std_dev=0.75))
    maninp4 = np.array(
        add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                        noise_std_dev=0.75))
    maninp5 = np.array(
        add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                        noise_std_dev=0.75))
    print('yes')
    maninp = np.concatenate(
        (maninp1, maninp2, maninp3, maninp4, maninp5))  # ,maninp6,maninp7,maninp8,maninp9,maninp10))

    return maninp


def batch_manipulate_rand(X_test, model_path, selected_class):
    maninp1 = np.array(
        add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                      noise_std_dev=0.75))
    maninp2 = np.array(
        add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                      noise_std_dev=0.75))
    maninp3 = np.array(
        add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                      noise_std_dev=0.75))
    maninp4 = np.array(
        add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                      noise_std_dev=0.75))
    maninp5 = np.array(
        add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                      noise_std_dev=0.75))

    maninp = np.concatenate((maninp1, maninp2, maninp3, maninp4, maninp5))

    return maninp


def batch_manipulate_frame(X_test):
    maninp1 = add_wn_frame(X_test, 1, noise_std_dev=0.75)
    maninp2 = add_wn_frame(X_test, 1, noise_std_dev=0.75)
    maninp3 = add_wn_frame(X_test, 1, noise_std_dev=0.75)
    maninp4 = add_wn_frame(X_test, 1, noise_std_dev=0.75)
    maninp5 = add_wn_frame(X_test, 1, noise_std_dev=0.75)

    maninp = np.concatenate((maninp1, maninp2, maninp3, maninp4, maninp5))

    return maninp


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def by_indices(outs, indices):
    return [[outs[i][0][indices]] for i in range(len(outs))]


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Coverage Analyzer for DNNs'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version", help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.")  # , required=True)
    # choices=['lenet1','lenet4', 'lenet5'], required=True)
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        or cifar10).", choices=["mnist", "cifar10"])  # , required=True)
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                        to measure coverage", choices=['cc', 'nc', 'kmnc',
                                                       'nbc', 'snac', 'tknc', 'ssc', 'lsa', 'dsa'])
    parser.add_argument("-C", "--class", help="the selected class", type=int)
    parser.add_argument("-Q", "--quantize", help="quantization granularity for \
                        combinatorial other_coverage_metrics.", type=int)
    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type=int)
    parser.add_argument("-KS", "--k_sections", help="number of sections used in \
                        k multisection other_coverage_metrics", type=int)
    parser.add_argument("-KN", "--k_neurons", help="number of neurons used in \
                        top k neuron other_coverage_metrics", type=int)
    parser.add_argument("-RN", "--rel_neurons", help="number of neurons considered\
                        as relevant in combinatorial other_coverage_metrics", type=int)
    parser.add_argument("-AT", "--act_threshold", help="a threshold value used\
                        to consider if a neuron is activated or not.", type=float)
    parser.add_argument("-R", "--repeat", help="index of the repeating. (for\
                        the cases where you need to run the same experiments \
                        multiple times)", type=int)
    parser.add_argument("-LOG", "--logfile", help="path to log file")
    parser.add_argument("-ADV", "--advtype", help="path to log file")
    parser.add_argument("-S", "--seed", help="seed to random", type=int)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    model_path = args['model'] if args['model'] else 'neural_networks/LeNet5'
    dataset = args['dataset'] if args['dataset'] else 'mnist'
    approach = args['approach'] if args['approach'] else 'cc'
    num_rel_neurons = args['rel_neurons'] if args['rel_neurons'] else 10
    act_threshold = args['act_threshold'] if args['act_threshold'] else 0
    top_k = args['k_neurons'] if args['k_neurons'] else 3
    k_sect = args['k_sections'] if args['k_sections'] else 1000
    selected_class = args['class'] if not args['class'] == None else -1  # ALL CLASSES
    logfile_name = args['logfile'] if args['logfile'] else 'result.log'
    quantization_granularity = args['quantize'] if args['quantize'] else 3
    adv_type = args['advtype'] if args['advtype'] else 'fgsm'
    seed = args['seed'] if args['seed'] else 1

    random.seed(seed)
    np.random.seed(seed)

    logfile = open(logfile_name, 'a')

    ####################
    # 0) Load MNIST or CIFAR10 data
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
        img_rows, img_cols = 28, 28
    else:
        X_train, Y_train, X_test, Y_test = load_CIFAR()
        img_rows, img_cols = 32, 32

    if not selected_class == -1:
        X_train, Y_train = filter_val_set(selected_class, X_train, Y_train)  # Get training input for selected_class
        X_test, Y_test = filter_val_set(selected_class, X_test, Y_test)  # Get testing input for selected_class

    ####################
    # 1) Setup the model
    model_name = model_path.split('/')[-1]

    try:
        json_file = open(model_path + '.json', 'r')  # Read Keras model parameters (stored in JSON file)
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

    # 2) Load necessary information
    trainable_layers = get_trainable_layers(model)
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))

    experiment_folder = 'experiments'

    # Investigate the penultimate layer
    subject_layer = args['layer'] if not args['layer'] == None else -1
    subject_layer = trainable_layers[subject_layer]

    skip_layers = [0]  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    print(skip_layers)

    ####################
    # 3) Analyze Coverages
    if approach == 'nc':

        fw = open('validation_mnist_nc.log', 'a')

        nc = NeuronCoverage(model, threshold=.75, skip_layers=skip_layers)  # SKIP ONLY INPUT AND FLATTEN LAYERS
        coverage, _, _, _, _ = nc.test(X_test)
        _, orig_acc = model.evaluate(X_test, Y_test)

        nc.set_measure_state(nc.get_measure_state())

        maninp = batch_manipulate_frame(X_test)
        # maninp = add_wn_frame(X_test, 1, noise_std_dev=0.4)
        frame_coverage, _, _, _, _ = nc.test(np.array(maninp))
        # _, frame_acc =  model.evaluate(np.array(maninp), Y_test)

        maninp = batch_manipulate_rel(X_test, model_path, selected_class)
        # maninp  = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta',
        #                          noise_std_dev=0.4)
        rel_coverage, _, _, _, _ = nc.test(np.array(maninp))
        # _, rel_acc = model.evaluate(np.array(maninp), Y_test)

        maninp = batch_manipulate_rand(X_test, model_path, selected_class)
        # maninp = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta',
        #                       noise_std_dev=0.4)
        rand_coverage, _, _, _, _ = nc.test(np.array(maninp))
        # _, rand_acc = model.evaluate(np.array(maninp), Y_test)

        res = {
            'model_name': model_name,
            'frame_coverage': frame_coverage,
            'frame_accuracy': 0,  # frame_acc,
            'orig_coverage': coverage,
            'orig_accuracy': 0,  # orig_acc,
            'rel_coverage': rel_coverage,
            'rel_accuracy': 0,  # rel_acc,
            'rand_coverage': rand_coverage,
            'rand_accuracy': 0,  # rand_acc,
            'seed': seed
        }

        fw.write(str(res))
        fw.write('\n')
        fw.close()

    elif approach == 'cc':

        X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model,
                                                                           X_train,
                                                                           Y_train)

        cc = CombCoverage(model, model_name, num_rel_neurons, selected_class,
                          subject_layer, X_train_corr, Y_train_corr)

        print("=====================")
        print("ORIGINAL")
        print("=====================")

        plt.ion()
        plt.imshow(X_test[0].reshape(28, 28))
        plt.savefig('mnist_orig.png')
        coverage, covered_combinations, orig_q = cc.test(X_test)
        _, orig_acc = model.evaluate(X_test, Y_test)
        print("Your test set's coverage is: ", coverage)
        print("Number of covered combinations: ", len(covered_combinations))
        print("Model evaluation: ", orig_acc)
        print("")

        cc.set_measure_state(covered_combinations)

        print("=====================")
        print("ADD WN FRAME")
        print("=====================")
        maninp = add_wn_frame(X_test, 1, 0.6)
        # plt.imshow(maninp[0].reshape(28,28))
        # plt.savefig('cifar_frm.png')
        frame_coverage, frame_covered_combinations, frame_q = cc.test(np.array(maninp))
        _, frame_acc = model.evaluate(np.array(maninp), Y_test)
        print("Your test set's coverage is: ", frame_coverage)
        print("Number of covered combinations: ", len(frame_covered_combinations))
        print("Model evaluation: ", model.evaluate(np.array(maninp), Y_test))
        print("")

        print("=====================")
        print("ADD WN RELEVANT PIXELS")
        print("=====================")
        maninp1 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp2 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp3 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp4 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp5 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp6 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp7 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp8 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp9 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                  noise_std_dev=0.75)
        maninp10 = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                                   noise_std_dev=0.75)

        maninp = np.concatenate(
            (maninp1, maninp2, maninp3, maninp4, maninp5, maninp6, maninp7, maninp8, maninp9, maninp10))
        # plt.imshow(maninp[0].reshape(28,28))
        # plt.savefig('mnist_rel.png')
        rel_coverage, rel_covered_combinations, rel_q = cc.test(np.array(maninp))
        _, rel_acc = model.evaluate(np.array(maninp1), Y_test)
        print("Your test set's coverage is: ", rel_coverage)
        print("Number of covered combinations: ", len(rel_covered_combinations))
        print("Model evaluation: ", model.evaluate(np.array(maninp1), Y_test))
        print("")

        print("=====================")
        print("ADD WN RANDOM PIXELS")
        print("=====================")
        maninp1 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp2 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp3 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp4 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp5 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp6 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp7 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp8 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp9 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                relevance_percentile=98)
        maninp10 = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=0.75,
                                 relevance_percentile=98)

        maninp = np.concatenate(
            (maninp1, maninp2, maninp3, maninp4, maninp5, maninp6, maninp7, maninp8, maninp9, maninp10))

        # plt.imshow(maninp[0].reshape(28,28))
        # plt.savefig('mnist_rand.png')
        rand_coverage, rand_covered_combinations, rand_q = cc.test(np.array(maninp))
        _, rand_acc = model.evaluate(np.array(maninp1), Y_test)
        print("Your test set's coverage is: ", rand_coverage)
        print("Number of covered combinations: ", len(rand_covered_combinations))
        print("Model evaluation: ", model.evaluate(np.array(maninp1), Y_test))
        print("")

        fw = open('validation_mnist_rc_rep5_with_seed_5test.log', 'a')
        res = {
            'model_name': model_name,
            'selected_class': selected_class,
            'subject_layer': subject_layer,
            'num_relevant_neurons': num_rel_neurons,
            'frame_coverage': frame_coverage,
            'frame_combinations': len(frame_covered_combinations),
            'frame_accuracy': frame_acc,
            'orig_coverage': coverage,
            'orig_combinations': len(covered_combinations),
            'orig_accuracy': orig_acc,
            'rel_coverage': rel_coverage,
            'rel_combinations': len(rel_covered_combinations),
            'rel_accuracy': rel_acc,
            'rand_coverage': rand_coverage,
            'rand_combinations': len(rand_covered_combinations),
            'rand_accuracy': rand_acc,
            'seed': seed
        }

        fw.write(str(res))
        fw.write('\n')
        fw.close()

    elif approach == 'kmnc' or approach == 'nbc' or approach == 'snac':
        res = {
            'model_name': model_name,
            'num_section': k_sect,
        }

        # SKIP ONLY INPUT AND FLATTEN LAYERS (USE GET_LAYER_OUTS_NEW)
        dg = DeepGaugePercentCoverage(model, k_sect, X_train, None, [0, 5])
        score = dg.test(X_test)
        _, orig_acc = model.evaluate(X_test, Y_test)

        kmnc_prcnt = score[0]
        nbc_prcnt = score[2]
        snac_prcnt = score[3]
        kmnc_cnt = score[-3]
        nbc_cnt = score[-2]
        snac_cnt = score[-1]

        res['orig_kmnc_prcnt'] = kmnc_prcnt
        res['orig_nbc_prcnt'] = nbc_prcnt
        res['orig_snac_prcnt'] = snac_prcnt
        res['orig_accuracy'] = orig_acc

        state = dg.get_measure_state()
        dg.set_measure_state(state)

        maninp = add_wn_frame(X_test, 1, noise_std_dev=0.75)
        score = dg.test(np.array(maninp))
        _, frame_acc = model.evaluate(np.array(maninp), Y_test)

        kmnc_prcnt = score[0]
        nbc_prcnt = score[2]
        snac_prcnt = score[3]
        kmnc_cnt = score[-3]
        nbc_cnt = score[-2]
        snac_cnt = score[-1]

        res['frame_kmnc_prcnt'] = kmnc_prcnt
        res['frame_nbc_prcnt'] = nbc_prcnt
        res['frame_snac_prcnt'] = snac_prcnt
        res['frame_accuracy'] = frame_acc

        maninp = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta',
                                 relevance_percentile=98, noise_std_dev=0.75)
        score = dg.test(np.array(maninp))
        _, rel_acc = model.evaluate(np.array(maninp), Y_test)

        kmnc_prcnt = score[0]
        nbc_prcnt = score[2]
        snac_prcnt = score[3]
        kmnc_cnt = score[-3]
        nbc_cnt = score[-2]
        snac_cnt = score[-1]

        res['rel_kmnc_prcnt'] = kmnc_prcnt
        res['rel_nbc_prcnt'] = nbc_prcnt
        res['rel_snac_prcnt'] = snac_prcnt
        res['rel_accuracy'] = rel_acc

        maninp = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta',
                               noise_std_dev=0.75, relevance_percentile=98)
        score = dg.test(np.array(maninp))
        _, rand_acc = model.evaluate(np.array(maninp), Y_test)

        kmnc_prcnt = score[0]
        nbc_prcnt = score[2]
        snac_prcnt = score[3]
        kmnc_cnt = score[-3]
        nbc_cnt = score[-2]
        snac_cnt = score[-1]

        res['rand_kmnc_prcnt'] = kmnc_prcnt,
        res['rand_nbc_prcnt'] = nbc_prcnt,
        res['rand_snac_prcnt'] = snac_prcnt,
        res['rand_accuracy'] = rand_acc

        f = open('validation_mnist_dg.log', 'a')

        f.write(str(res))
        f.write('\n')
        f.close()

    elif approach == 'tknc':
        # SKIP ONLY INPUT AND FLATTEN LAYERS (USE GET_LAYER_OUTS_NEW)
        dg = DeepGaugeLayerLevelCoverage(model, top_k, skip_layers=[0, 5])
        orig_coverage, _, _, _, _, _ = dg.test(X_test)
        _, orig_acc = model.evaluate(X_test, Y_test)

        maninp = add_wn_frame(X_test, 1, noise_std_dev=0.75)
        frame_coverage, _, _, _, _, _ = dg.test(np.array(maninp))
        _, frame_acc = model.evaluate(np.array(maninp), Y_test)

        maninp = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta',
                                 relevance_percentile=98, noise_std_dev=0.75)
        rel_coverage, _, _, _, _, _ = dg.test(np.array(maninp))
        _, rel_acc = model.evaluate(np.array(maninp), Y_test)

        maninp = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta',
                               noise_std_dev=0.75, relevance_percentile=98)
        rand_coverage, _, _, _, _, _ = dg.test(np.array(maninp))
        _, rand_acc = model.evaluate(np.array(maninp), Y_test)

        f = open('validation_mnist_tkn.log', 'a')
        res = {
            'model_name': model_name,
            'frame_coverage': frame_coverage,
            'frame_accuracy': frame_acc,
            'orig_coverage': orig_coverage,
            'orig_accuracy': orig_acc,
            'rel_coverage': rel_coverage,
            'rel_accuracy': rel_acc,
            'rand_coverage': rand_coverage,
            'rand_accuracy': rand_acc,
            'seed': seed
        }

        f.write(str(res))
        f.write('\n')
        f.close()

    elif approach == 'ssc':

        ss = SSCover(model, skip_layers=non_trainable_layers)
        score = ss.test(X_test)

        if os.path.isfile('%s/%s_%d_%s_adversarial_dataset.h5' % (experiment_folder, model_name, selected_class, adv_type)):
            X_adv = load_data('%s/%s_%d_%s_adversarial' % (experiment_folder, model_name, selected_class, adv_type))
        else:
            X_adv = generate_adversarial(list(X_test), adv_type, model)
            save_data(X_adv, '%s/%s_%d_%s_adversarial' % (experiment_folder, model_name, selected_class, adv_type))

        score = ss.test(X_adv)

    elif approach == 'lsa' or approach == 'dsa':

        if approach == 'lsa':
            upper_bound = 2000
        elif approach == 'dsa':
            upper_bound = 2

        if model_name == 'LeNet4':
            layer_names = [model.layers[-2].name]
        elif model_name == 'LeNet5' or model_name == 'LeNet1':
            layer_names = [model.layers[-3].name]

        sa = SurpriseAdequacy([], model, X_train, layer_names, upper_bound, dataset)

        orig_coverage = sa.test(X_test, 'orig', approach)
        _, orig_acc = model.evaluate(X_test, Y_test)

        # sa.set_measure_state(sa.get_measure_state())

        maninp = add_wn_frame(X_test, 1, noise_std_dev=0.75)
        frame_coverage = sa.test(np.concatenate((np.array(maninp), X_test), axis=0), 'frame', approach)
        _, frame_acc = model.evaluate(np.array(maninp), Y_test)

        maninp = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta',
                                 relevance_percentile=98, noise_std_dev=0.75)
        rel_coverage = sa.test(np.concatenate((np.array(maninp), X_test), axis=0), 'wn_rel', approach)
        _, rel_acc = model.evaluate(np.array(maninp), Y_test)

        maninp = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta',
                               noise_std_dev=0.75, relevance_percentile=98)
        rand_coverage = sa.test(np.concatenate((np.array(maninp), X_test), axis=0), 'wn_rand', approach)
        _, rand_acc = model.evaluate(np.array(maninp), Y_test)

        f = open('validation_mnist_sa.log', 'a')
        res = {
            'model_name': model_name,
            'approach': approach,
            'frame_coverage': frame_coverage,
            'frame_accuracy': frame_acc,
            'orig_coverage': orig_coverage,
            'orig_accuracy': orig_acc,
            'rel_coverage': rel_coverage,
            'rel_accuracy': rel_acc,
            'rand_coverage': rand_coverage,
            'rand_accuracy': rand_acc,
            'seed': seed
        }

        f.write(str(res))
        f.write('\n')
        f.close()

    logfile.close()
