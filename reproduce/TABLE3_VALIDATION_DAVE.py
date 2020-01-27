from reproduce.manipulate_dave import add_wn_frame, add_white_noise, add_wn_random
from utils import load_driving_data, load_dave_model
from utils import get_trainable_layers
from utils import save_data, load_data
from utils import generate_adversarial
from utils import preprocess_image
from coverages.idc import CombCoverage
from coverages.neuron_cov import NeuronCoverage
from coverages.tkn import DeepGaugeLayerLevelCoverage
from coverages.kmn import DeepGaugePercentCoverage
from coverages.ss import SSCover
from coverages.sa import SurpriseAdequacy

import os
import random
import argparse
import datetime
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

__version__ = 0.9


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
    parser.add_argument("-LOG", "--logfile", help="path to log file")
    parser.add_argument("-ADV", "--advtype", help="path to log file")
    parser.add_argument("-S", "--seed", help="seed t0 random", type=int)

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

    logfile = open(logfile_name, 'a')

    random.seed(seed)
    np.random.seed(seed)

    ####################
    # 0) Load Driving Data

    X_all = []
    X_paths, Ys = load_driving_data()
    for xp in X_paths:
        X_all.append(preprocess_image(xp)[0])

    print("LOAD DONE")

    # TODO: Check numbers here
    X_train = np.array(X_all)
    X_test = np.array(X_all[4000:])
    Y_train = np.array(Ys)
    Y_test = np.array(Ys[4000:])

    ####################
    # 1) Setup the model
    model_name = model_path.split('/')[-1]
    model = load_dave_model()

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

    ####################
    # 3) Analyze Coverages
    if approach == 'nc':

        fw = open('validation_dave_nc.log', 'a')

        nc = NeuronCoverage(model, threshold=.75, skip_layers=skip_layers)  # SKIP ONLY INPUT AND FLATTEN LAYERS
        coverage, _, _, _, _ = nc.test(X_test)
        orig_err = model.evaluate(X_test, Y_test)

        nc.set_measure_state(nc.get_measure_state())

        maninp = add_wn_frame(X_test, 1, noise_std_dev=0.75)
        frame_coverage, _, _, _, _ = nc.test(np.array(maninp))
        frame_err = model.evaluate(np.array(maninp), Y_test)

        maninp = add_white_noise(X_test, model_path, selected_class,
                                 lrpmethod='simple', relevance_percentile=98,
                                 noise_std_dev=0.75)
        rel_coverage, _, _, _, _ = nc.test(np.array(maninp))
        rel_err = model.evaluate(np.array(maninp), Y_test)

        maninp = add_wn_random(X_test, model_path, selected_class, lrpmethod='simple',
                               noise_std_dev=0.75, relevance_percentile=98)
        rand_coverage, _, _, _, _ = nc.test(np.array(maninp))
        rand_err = model.evaluate(np.array(maninp), Y_test)

        res = {
            'model_name': model_name,
            'frame_coverage': frame_coverage,
            'frame_accuracy': frame_err,
            'orig_coverage': coverage,
            'orig_accuracy': orig_err,
            'rel_coverage': rel_coverage,
            'rel_accuracy': rel_err,
            'rand_coverage': rand_coverage,
            'rand_accuracy': rand_err,
            'seed': seed
        }

        fw.write(str(res))
        fw.write('\n')
        fw.close()

    elif approach == 'cc':

        cc = CombCoverage(model, model_name, num_rel_neurons, selected_class,
                          subject_layer, X_train, Y_train)

        plt.ion()
        plt.imshow(X_all[0].reshape(100, 100, 3))
        plt.savefig('dave_orig.png')
        coverage, covered_combinations, orig_q = cc.test(X_test)

        orig_mse = model.evaluate(np.array(X_all), np.array(Ys))

        print("ORIG COV:")
        print(coverage)

        cc.set_measure_state(covered_combinations)

        maninp = add_wn_frame(np.array(X_test), 1, 192)
        plt.imshow(maninp[0].reshape(100, 100, 3))
        plt.savefig('dave_frm.png')
        frame_coverage, frame_covered_combinations, frame_q = cc.test(np.array(maninp))
        frame_mse = model.evaluate(np.array(maninp), Y_test)
        print("FRAME COV:")
        print("Your test set's coverage is: ", frame_coverage)

        s = datetime.datetime.now()
        print("=====================")
        print("ADD WN RELEVANT PIXELS")
        print("=====================")
        maninp1 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp2 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp3 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp4 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp5 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp6 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp7 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp8 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp9 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                           relevance_percentile=98, noise_std_dev=192))
        maninp10 = np.array(add_white_noise(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta',
                                            relevance_percentile=98, noise_std_dev=192))

        maninp = np.concatenate(
            (maninp1, maninp2, maninp3, maninp4, maninp5, maninp6, maninp7, maninp8, maninp9, maninp10))

        # plt.imshow(maninp[0].reshape(100,100,3))
        # plt.savefig('dave_rel.png')

        rel_coverage, rel_covered_combinations, rel_q = cc.test(np.array(maninp))
        rel_mse = model.evaluate(np.array(maninp1), Y_test)
        print("Your test set's coverage is: ", rel_coverage)
        print("")
        e = datetime.datetime.now()
        print(e - s)

        s = datetime.datetime.now()
        print("=====================")
        print("ADD WN RANDOM PIXELS")
        print("=====================")
        maninp1 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp2 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp3 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp4 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp5 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp6 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp7 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp8 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp9 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))
        maninp10 = np.array(
            add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', relevance_percentile=98,
                          noise_std_dev=192))

        maninp = np.concatenate(
            (maninp1, maninp2, maninp3, maninp4, maninp5, maninp6, maninp7, maninp8, maninp9, maninp10))

        # maninp = add_wn_random(np.array(X_test), model_path, selected_class, lrpmethod='alphabeta', noise_std_dev=192, relevance_percentile=98)

        # plt.imshow(maninp[0].reshape(100,100,3))
        # plt.savefig('dave_rand.png')
        rand_coverage, rand_covered_combinations, rand_q = cc.test(np.array(maninp))
        rand_mse = model.evaluate(np.array(maninp1), Y_test)
        print("Your test set's coverage is: ", rand_coverage)
        print("")
        e = datetime.datetime.now()
        print(e - s)

        fw = open('validation_dave_rc_rep5_with_seed_5test.log', 'a')
        res = {
            'model_name': model_name,
            'selected_class': selected_class,
            'subject_layer': subject_layer,
            'num_relevant_neurons': num_rel_neurons,
            'frame_coverage': frame_coverage,
            'frame_combinations': len(frame_covered_combinations),
            'frame_error': frame_mse,
            'orig_coverage': coverage,
            'orig_combinations': len(covered_combinations),
            'orig_error': orig_mse,
            'rel_coverage': rel_coverage,
            'rel_combinations': len(rel_covered_combinations),
            'rel_error': rel_mse,
            'rand_coverage': rand_coverage,
            'rand_combinations': len(rand_covered_combinations),
            'rand_errror': rand_mse,
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

        maninp = add_wn_frame(X_test, 1, noise_std_dev=0.4)
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
                                 noise_std_dev=0.4)
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
                               noise_std_dev=0.4)
        score = dg.test(np.array(maninp))
        _, rand_acc = model.evaluate(np.array(maninp), Y_test)

        kmnc_prcnt = score[0]
        nbc_prcnt = score[2]
        snac_prcnt = score[3]
        kmnc_cnt = score[-3]
        nbc_cnt = score[-2]
        snac_cnt = score[-1]

        res['rand_kmnc_prcnt'] = kmnc_prcnt
        res['rand_nbc_prcnt'] = nbc_prcnt
        res['rand_snac_prcnt'] = snac_prcnt
        res['rand_accuracy'] = rand_acc

        res['seed'] = seed

        f = open('validation_dave_dg.log', 'a')

        f.write(str(res))
        f.write('\n')
        f.close()

    elif approach == 'tknc':
        dg = DeepGaugeLayerLevelCoverage(model, top_k, skip_layers=[0, 5])
        orig_coverage, _, _, _, _, _ = dg.test(X_test)
        orig_err = model.evaluate(X_test, Y_test)

        maninp = add_wn_frame(X_test, 1, noise_std_dev=0.4)
        frame_coverage, _, _, _, _, _ = dg.test(np.array(maninp))
        frame_err = model.evaluate(np.array(maninp), Y_test)

        maninp = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta',
                                 noise_std_dev=0.4)
        rel_coverage, _, _, _, _, _ = dg.test(np.array(maninp))
        rel_err = model.evaluate(np.array(maninp), Y_test)

        maninp = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta',
                               noise_std_dev=0.4)
        rand_coverage, _, _, _, _, _ = dg.test(np.array(maninp))
        rand_err = model.evaluate(np.array(maninp), Y_test)

        f = open('validation_dave_tkn.log', 'a')
        res = {
            'model_name': model_name,
            'frame_coverage': frame_coverage,
            'frame_accuracy': frame_err,
            'orig_coverage': orig_coverage,
            'orig_accuracy': orig_err,
            'rel_coverage': rel_coverage,
            'rel_accuracy': rel_err,
            'rand_coverage': rand_coverage,
            'rand_accuracy': rand_err,
            'seed': seed
        }

        f.write(str(res))
        f.write('\n')
        f.close()

    elif approach == 'ssc':
        ss = SSCover(model, skip_layers=non_trainable_layers)
        score = ss.test(X_test)

        if os.path.isfile('%s/%s_%d_%s_adversarial_dataset.h5' % (experiment_folder, \
                                                                  model_name, selected_class, adv_type)):
            X_adv = load_data('%s/%s_%d_%s_adversarial' % (experiment_folder, \
                                                           model_name, selected_class, adv_type))
        else:
            X_adv = generate_adversarial(list(X_test), adv_type, model)
            save_data(X_adv, '%s/%s_%d_%s_adversarial' % (experiment_folder, \
                                                          model_name, selected_class, adv_type))

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
        orig_err = model.evaluate(X_test, Y_test)

        maninp = add_wn_frame(X_test, 1, noise_std_dev=0.4)
        frame_coverage = sa.test(np.concatenate((np.array(maninp), X_test), axis=0), 'frame', approach)
        frame_err = model.evaluate(np.array(maninp), Y_test)

        maninp = add_white_noise(X_test, model_path, selected_class, lrpmethod='alphabeta',
                                 noise_std_dev=0.4)
        rel_coverage = sa.test(np.concatenate((np.array(maninp), X_test), axis=0), 'wn_rel', approach)
        rel_err = model.evaluate(np.array(maninp), Y_test)

        maninp = add_wn_random(X_test, model_path, selected_class, lrpmethod='alphabeta',
                               noise_std_dev=0.4)
        rand_coverage = sa.test(np.concatenate((np.array(maninp), X_test), axis=0), 'wn_rand', approach)
        rand_err = model.evaluate(np.array(maninp), Y_test)

        f = open('validation_dave_sa.log', 'a')
        res = {
            'model_name': model_name,
            'approach': approach,
            'frame_coverage': frame_coverage,
            'frame_accuracy': frame_err,
            'orig_coverage': orig_coverage,
            'orig_accuracy': orig_err,
            'rel_coverage': rel_coverage,
            'rel_accuracy': rel_err,
            'rand_coverage': rand_coverage,
            'rand_accuracy': rand_err,
            'seed': seed
        }

        f.write(str(res))
        f.write('\n')
        f.close()

    logfile.close()
