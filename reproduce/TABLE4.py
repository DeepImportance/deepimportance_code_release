from keras.models import model_from_json, load_model
from reproduce.manipulate import add_wn_all, add_wn_frame, add_white_noise, add_wn_random
from utils import load_MNIST, load_CIFAR
from utils import filter_val_set, get_trainable_layers
from utils import save_data, load_data
from utils import generate_adversarial, filter_correct_classifications
from coverages.idc import CombCoverage
from coverages.neuron_cov import NeuronCoverage
from coverages.tkn import DeepGaugeLayerLevelCoverage
from coverages.kmn import DeepGaugePercentCoverage
from coverages.ss import SSCover
from coverages.sa import SurpriseAdequacy
import os
import argparse
import numpy as np

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
    parser.add_argument("-R", "--repeat", help="index of the repeating. (for\
                        the cases where you need to run the same experiments \
                        multiple times)", type=int)
    parser.add_argument("-LOG", "--logfile", help="path to log file")
    parser.add_argument("-ADV", "--advtype", help="adversarial input generation method")

    # parse command-line arguments

    # parse command-line arguments
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
    repeat = args['repeat'] if args['repeat'] else 1
    logfile_name = args['logfile'] if args['logfile'] else 'result.log'
    quantization_granularity = args['quantize'] if args['quantize'] else 3
    adv_type = args['advtype'] if args['advtype'] else 'fgsm'

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

        nc = NeuronCoverage(model, threshold=.75, skip_layers=skip_layers)  # SKIP ONLY INPUT AND FLATTEN LAYERS
        orig_coverage, _, _, _, _ = nc.test(X_test)

        nc.set_measure_state(nc.get_measure_state())

        if os.path.isfile('%s/%s_%d_%s_adversarial_dataset.h5' % (experiment_folder,
                                                                  model_name,
                                                                  selected_class,
                                                                  adv_type)):
            X_adv = load_data('%s/%s_%d_%s_adversarial' % (experiment_folder,
                                                           model_name,
                                                           selected_class, adv_type))
        else:
            if adv_type == 'gauss':
                X_adv = add_wn_all(X_test)
                X_adv = np.array(X_adv)
            else:
                X_adv = generate_adversarial(list(X_test), adv_type, model)
                save_data(X_adv, '%s/%s_%d_%s_adversarial' % (experiment_folder,
                                                              model_name,
                                                              selected_class, adv_type))

        adv_coverage, _, _, _, _ = nc.test(X_adv)

        # _, adv_acc = model.evaluate(X_adv, Y_test)

        fw = open('effectiveness_mnist_nc.log', 'a')
        res = {
            'model_name': model_name,
            'adv_type': adv_type,
            # 'selected_class': selected_class,
            # 'subject_layer': subject_layer,
            # 'num_relevant_neurons': num_rel_neurons,
            # 'q_granul': quantization_granularity,
            'orig_coverage': orig_coverage,
            # 'orig_combinations': len(orig_covered_combinations),
            'orig_accuracy': 1,
            'adv_coverage': adv_coverage,
            # 'adv_combinations': len(adv_covered_combinations),
            'adv_accuracy': 0
        }

        fw.write(str(res))
        fw.write('\n')
        fw.close()

    elif approach == 'cc':
        X_all = np.load('cwres/advLeNet1.npy')

        X_adv, Y_ = filter_val_set(selected_class, X_all, Y_test)

        X_test, Y_test = filter_val_set(selected_class, X_test, Y_test)

        X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model,
                                                                           X_train,
                                                                           Y_train)

        cc = CombCoverage(model, model_name, num_rel_neurons, selected_class,
                          subject_layer, X_train_corr, Y_train_corr)

        orig_coverage, orig_covered_combinations, _ = cc.test(X_test)
        _, orig_acc = model.evaluate(X_test, Y_test)

        cc.set_measure_state(orig_covered_combinations)
        adv_coverage, adv_covered_combinations, _ = cc.test(X_adv)
        _, adv_acc = 0, 0  # model.evaluate(X_adv, Y_test)

        fw = open('effectiveness_mnist_cc_cw.log', 'a')
        res = {
            'model_name': model_name,
            'adv_type': adv_type,
            'selected_class': selected_class,
            'subject_layer': subject_layer,
            'num_relevant_neurons': num_rel_neurons,
            'q_granul': quantization_granularity,
            'orig_coverage': orig_coverage,
            'orig_combinations': len(orig_covered_combinations),
            'orig_accuracy': orig_acc,
            'adv_coverage': adv_coverage,
            'adv_combinations': len(adv_covered_combinations),
            'adv_accuracy': adv_acc
        }

        fw.write(str(res))
        fw.write('\n')
        fw.close()

    elif approach == 'kmnc' or approach == 'nbc' or approach == 'snac':
        # SKIP ONLY INPUT AND FLATTEN LAYERS (USE GET_LAYER_OUTS_NEW)
        dg = DeepGaugePercentCoverage(model, k_sect, X_train, None, skip_layers)
        score = dg.test(X_test)

        orig_kmnc_prcnt = score[0]
        orig_nbc_prcnt = score[2]
        orig_snac_prcnt = score[3]

        _, orig_acc = model.evaluate(X_test, Y_test)

        state = dg.get_measure_state()
        dg.set_measure_state(state)

        if os.path.isfile('%s/%s_%d_%s_adversarial_dataset.h5' % (experiment_folder,
                                                                  model_name,
                                                                  selected_class,
                                                                  adv_type)):
            X_adv = load_data('%s/%s_%d_%s_adversarial' % (experiment_folder,
                                                           model_name,
                                                           selected_class, adv_type))
        else:
            if adv_type == 'gauss':
                X_adv = add_wn_all(X_test)
                X_adv = np.array(X_adv)
            else:
                X_adv = generate_adversarial(list(X_test), adv_type, model)
                save_data(X_adv, '%s/%s_%d_%s_adversarial' % (experiment_folder,
                                                              model_name,
                                                              selected_class, adv_type))

        score = dg.test(X_adv)

        adv_kmnc_prcnt = score[0]
        adv_nbc_prcnt = score[2]
        adv_snac_prcnt = score[3]

        f = open('effectiveness_dg.log', 'a')
        res = {
            'model_name': model_name,
            'num_section': k_sect,
            'adv_type': adv_type,
            'orig_kmnc_prcnt': orig_kmnc_prcnt,
            'orig_nbc_prcnt': orig_nbc_prcnt,
            'orig_snac_prcnt': orig_snac_prcnt,
            'adv_kmnc_prcnt': adv_kmnc_prcnt,
            'adv_nbc_prcnt': adv_nbc_prcnt,
            'adv_snac_prcnt': adv_snac_prcnt,
            'orig_accuracy': orig_acc
        }

        f.write(str(res))
        f.write('\n')
        f.close()

    elif approach == 'tknc':
        # SKIP ONLY INPUT AND FLATTEN LAYERS (USE GET_LAYER_OUTS_NEW)
        dg = DeepGaugeLayerLevelCoverage(model, top_k, skip_layers=skip_layers)
        score = dg.test(X_test)
        orig_coverage = score[0]
        _, orig_acc = model.evaluate(X_test, Y_test)

        dg.set_measure_state(dg.get_measure_state())

        if os.path.isfile('%s/%s_%d_%s_adversarial_dataset.h5' % (experiment_folder,
                                                                  model_name,
                                                                  selected_class,
                                                                  adv_type)):
            X_adv = load_data('%s/%s_%d_%s_adversarial' % (experiment_folder,
                                                           model_name,
                                                           selected_class, adv_type))
        else:
            if adv_type == 'gauss':
                X_adv = add_wn_all(X_test)
                X_adv = np.array(X_adv)
            else:
                X_adv = generate_adversarial(list(X_test), adv_type, model)
                save_data(X_adv, '%s/%s_%d_%s_adversarial' % (experiment_folder,
                                                              model_name,
                                                              selected_class, adv_type))

        score = dg.test(X_adv)
        adv_coverage = score[0]

        fw = open('effectiveness_tkn.log', 'a')
        res = {
            'model_name': model_name,
            'adv_type': adv_type,
            'orig_coverage': orig_coverage,
            'orig_accuracy': orig_acc,
            'adv_coverage': adv_coverage,
            'adv_accuracy': 0
        }

        fw.write(str(res))
        fw.write('\n')
        fw.close()

    elif approach == 'ssc':

        ss = SSCover(model, skip_layers=skip_layers)
        score = ss.test(X_test[:10])
        _, orig_acc = model.evaluate(X_test, Y_test)
        orig_coverage = score[0]
        print(orig_coverage)
        exit()

        if os.path.isfile('%s/%s_%d_%s_adversarial_dataset.h5' % (experiment_folder,
                                                                  model_name,
                                                                  selected_class,
                                                                  adv_type)):
            X_adv = load_data('%s/%s_%d_%s_adversarial' % (experiment_folder,
                                                           model_name,
                                                           selected_class,
                                                           adv_type))
        else:
            if adv_type == 'gauss':
                X_adv = add_wn_all(X_test)
                X_adv = np.array(X_adv)
            else:
                X_adv = generate_adversarial(list(X_test), adv_type, model)
                save_data(X_adv, '%s/%s_%d_%s_adversarial' % (experiment_folder,
                                                              model_name,
                                                              selected_class,
                                                              adv_type))

        score = ss.test(X_adv)
        _, adv_acc = model.evaluate(X_adv, Y_test)
        adv_coverage = score[0]

        fw = open('effectiveness_ssc.log', 'a')

        res = {
            'model_name': model_name,
            'adv_type': adv_type,
            'orig_coverage': orig_coverage,
            'orig_accuracy': orig_acc,
            'adv_coverage': adv_coverage,
            'adv_accuracy': adv_acc
        }

        fw.write(str(res))
        fw.write('\n')
        fw.close()

    elif approach == 'lsa' or approach == 'dsa':

        if approach == 'lsa':
            upper_bound = 2000
        elif approach == 'dsa':
            upper_bound = 2

        if model_name == 'LeNet4':
            layer_names = [model.layers[-2].name]
        elif model_name == 'LeNet5' or model_name == 'LeNet1':
            layer_names = [model.layers[-3].name]

        X_adv = np.load('cwres/advLeNet1.npy')

        sa = SurpriseAdequacy([], model, X_train, layer_names, upper_bound, dataset)

        orig_coverage = sa.test(X_test, 'orig_effec', approach)

        sa.set_measure_state(sa.get_measure_state())

        '''
        if os.path.isfile('%s/%s_%d_%s_adversarial_dataset.h5' %(experiment_folder,
                                                                 model_name,
                                                                 selected_class,
                                                                 adv_type)):
            X_adv = load_data('%s/%s_%d_%s_adversarial' %(experiment_folder,
                                                          model_name,
                                                          selected_class, adv_type))
        else:
            if adv_type == 'gauss':
                X_adv = add_wn_all(X_test)
                X_adv = np.array(X_adv)
            else:
                X_adv = generate_adversarial(list(X_test), adv_type, model)
                save_data(X_adv, '%s/%s_%d_%s_adversarial' %(experiment_folder,
                                                             model_name,
                                                             selected_class, adv_type))
        '''

        adv_coverage = sa.test(X_adv, adv_type + '_effec', approach)
        # adv_coverage = sa.test(np.concatenate((X_adv, X_test),axis=0), approach)

        fw = open('effectiveness_mnist_sa_cw.log', 'a')
        res = {
            'model_name': model_name,
            'adv_type': adv_type,
            'approach': approach,
            'orig_coverage': orig_coverage,
            'orig_accuracy': 1,
            'adv_coverage': adv_coverage,
            'adv_accuracy': 0
        }

        fw.write(str(res))
        fw.write('\n')
        fw.close()

    logfile.close()
