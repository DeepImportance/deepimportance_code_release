import os
import itertools
import numpy as np
from random import choice, sample
from lrp_toolbox.model_io import read
from utils import find_relevant_pixels
from utils import save_relevant_pixels, load_relevant_pixels
from utils import load_data, save_data

exp_folder = 'experiments'


def add_wn_all(inputs, noise_std_dev=0.3):
    manpltd_inputs = []
    for idx, inp in enumerate(inputs):
        minp = np.array([row[:] for row in inp])
        for (i, j, k) in itertools.product(range(inp.shape[0]),
                                           range(inp.shape[1]),
                                           range(inp.shape[2])):
            prtrb = np.random.normal(0, noise_std_dev)
            if minp[i][j][k] + prtrb > 1:
                minp[i][j][k] = 1
            elif minp[i][j][k] + prtrb < 0:
                minp[i][j][k] = 0
            else:
                minp[i][j][k] += prtrb

        manpltd_inputs.append(minp)
    return manpltd_inputs


def add_wn_frame(inputs, frame_thickness, noise_std_dev=0.3):
    '''
    if os.path.isfile('%s/%d_%f_%s.h5' %
                      (exp_folder, len(inputs), noise_std_dev, 'wn_frame')):
        manpltd_inputs = load_data(
            '%s/%d_%f_%s.h5' %
            (exp_folder, len(inputs), noise_std_dev, 'wn_frame'))
    else:
    '''

    manpltd_inputs = []
    img_width = inputs.shape[1]
    img_length = inputs.shape[2]

    for inp in inputs:
        minp = np.array([row[:] for row in inp])
        num_change = 0
        for i in range(img_width):
            for j in range(img_length):
                if (i < frame_thickness or j < frame_thickness or i >
                        (img_length - frame_thickness - 1) or j >
                        (img_width - frame_thickness - 1)
                        ) and num_change < 16:  # 2% of num pixels
                    minp[i][j] += abs(np.random.normal(0, noise_std_dev))
                    num_change += 1

        manpltd_inputs.append(minp)
    '''
        save_data(
            manpltd_inputs, '%s/%d_%f_%s.h5' %
            (exp_folder, len(inputs), noise_std_dev, 'wn_frame'))
    '''
    return manpltd_inputs


def add_white_noise(inputs,
                    model_path,
                    selected_class,
                    lrpmethod='simple',
                    relevance_percentile=90,
                    noise_std_dev=0.3):
    model_name = model_path.split('/')[1]
    fname = '%s/%s_%s_%d_%d' % (exp_folder, model_name, lrpmethod,
                                selected_class, relevance_percentile)

    '''
    if os.path.isfile(fname + 'wn_rel.h5'):
        manpltd_inputs = load_data(fname + 'wn_rel.h5')
        print("LOADED")
    else:
    '''
    try:
        all_relevant_pixels = load_relevant_pixels(fname)
    except:
        all_relevant_pixels = find_relevant_pixels(inputs, model_path,
                                                    lrpmethod,
                                                    relevance_percentile)
        save_relevant_pixels(fname, all_relevant_pixels)
        print("NOT FOUND!")

    manpltd_inputs = []
    avg_num_relevant_pixels = 0
    for idx, inp in enumerate(inputs):
        minp = np.array([row[:] for row in inp])
        relevant_pixels = all_relevant_pixels[idx]
        for i, j in zip(relevant_pixels[0], relevant_pixels[1]):
            prtrb = np.random.normal(0, noise_std_dev)
            if minp[i][j] + prtrb > 1:
                minp[i][j] = 1
            elif minp[i][j] + prtrb < 0:
                minp[i][j] = 0
            else:
                minp[i][j] += prtrb

        manpltd_inputs.append(minp)

        #save_data(manpltd_inputs, fname + 'wn_rel.h5')

    return manpltd_inputs


def add_wn_random(inputs,
                  model_path,
                  selected_class,
                  lrpmethod='simple',
                  noise_std_dev=0.3,
                  relevance_percentile=90):

    model_name = model_path.split('/')[1]

    fname = 'experiments/%s_%s_%d_%d' % (model_name, lrpmethod, selected_class,
                                         relevance_percentile)
    '''
    if os.path.isfile(fname + 'wn_rand.h5'):
        manpltd_inputs = load_data(fname + 'wn_rand.h5')
    else:
    '''

    try:
        all_relevant_pixels = load_relevant_pixels(fname)
    except:
        all_relevant_pixels = find_relevant_pixels(inputs, model_path,
                                                    lrpmethod,
                                                    relevance_percentile)
        save_relevant_pixels(fname, all_relevant_pixels)
        print("NOT FOUND!")

    manpltd_inputs = []

    for idx, inp in enumerate(inputs):

        relevant_pixels = all_relevant_pixels[idx]
        relevant_pixels_merged = [
            (i, j) for i, j in zip(relevant_pixels[0], relevant_pixels[1])
        ]

        data = [elem for elem in itertools.product(*[range(28),range(28)])]
        data = list(set(data)-set(relevant_pixels_merged))

#        data = [
#            [i, j] for i in range(28) for j in range(28)
#            if [i, j] not in relevant_pixels_merged
#        ]  # if i not in relevant_pixels[0] or j not in relevant_pixels[1]]

        random_pixels = sample(data, len(relevant_pixels[0]))

        minp = np.array([row[:] for row in inp])
        for [i, j
                ] in random_pixels:  # zip(random_pixels[0], random_pixels[1]):
            prtrb = np.random.normal(0, noise_std_dev)
            if minp[i][j] + prtrb > 1:
                minp[i][j] = 1
            elif minp[i][j] + prtrb < 0:
                minp[i][j] = 0
            else:
                minp[i][j] += prtrb

        manpltd_inputs.append(minp)

        #save_data(manpltd_inputs, fname + 'wn_rand.h5')

    return manpltd_inputs


def block_frame(inputs, frame_thickness):
    manpltd_inputs = []
    img_width = inputs.shape[1]
    img_length = inputs.shape[2]

    #minputs = inputs[:,:,:,:]
    # print(id(inputs))
    # print(id(minputs))

    for inp in inputs:
        minp = np.array([row[:] for row in inp])  # np.zeros(inp.shape)
        for i in range(img_width):
            for j in range(img_length):
                # or j < frame_thickness: or i > (img_length - frame_thickness -1) or j > (img_width - frame_thickness -1):
                if i < frame_thickness:
                    minp[i][j] = 1
                # else:
                #    minp[i][j]=inp[i][j]

        manpltd_inputs.append(minp)

    return manpltd_inputs


def block_relevant_pixels(inputs,
                    model_path,
                    selected_class,
                    lrpmethod='simple',
                    relevance_percentile=90):

    model_name = model_path.split('/')[1]
    fname = '%s/%s_%s_%d_%d' % (exp_folder, model_name, lrpmethod,
                                selected_class, relevance_percentile)

    try:
        all_relevant_pixels = load_relevant_pixels(fname)
    except:
        all_relevant_pixels = find_relevant_pixels(inputs, model_path,
                                                    lrpmethod,
                                                    relevance_percentile)
        save_relevant_pixels(fname, all_relevant_pixels)
        print("NOT FOUND!")

    manpltd_inputs = []

    for idx,inp in enumerate(inputs):
        relevant_pixels = all_relevant_pixels[idx]
        minp = inp.copy()
        for i, j in zip(relevant_pixels[0], relevant_pixels[1]):
            minp[i][j] = 0
            # PIXEL FLIPPING
            #if abs(minp[i][j] - 1) > minp[i][j]:
            #    minp[i][j] = 1
            #else:
            #    minp[i][j] = 0

        manpltd_inputs.append(minp)


    return manpltd_inputs


def block_random_pixels(inputs, num_pixels):

    manpltd_inputs = []

    for inp in inputs:

        random_pixels = np.random.randint(0, 28, size=(2, num_pixels))

        #rand_x = choice([])

        minp = np.array([row[:] for row in inp])
        for i, j in zip(random_pixels[0], random_pixels[1]):
            minp[i][j] = 0
            # PIXEL FLIPPING
            #if abs(minp[i][j] - 1) > minp[i][j]:
            #    minp[i][j] = 1
            #else:
            #    minp[i][j] = 0

        manpltd_inputs.append(minp)

    return manpltd_inputs
