import itertools
import datetime
import numpy as np
import matplotlib.pyplot as plt
from lrp_toolbox.model_io import write, read
from random import choice, sample
from utils import find_relevant_pixels
from utils import save_relevant_pixels, load_relevant_pixels

np.random.seed( 321 )


def add_wn_frame(inputs, frame_thickness, noise_std_dev=50):
    manpltd_inputs = []
    img_width  = inputs.shape[1]
    img_length = inputs.shape[2]
    img_depth  = inputs.shape[3]

    for inp in inputs:
        minp = np.array([row[:] for row in inp])
        num_change = 0
        for i in range(img_width):
            for j in range(img_length):
                for k in range(img_depth):    #NUM_CHANGE -> 2% * 3 CHANNELS
                    if (i < frame_thickness or j < frame_thickness or i > (img_length - frame_thickness -1) or j > (img_width - frame_thickness -1)) and num_change < 600:
                        prtrb = np.random.normal(0,noise_std_dev)
                        if minp[i][j][k] + prtrb > 255: minp[i][j][k] = 255 #maxed out
                        elif minp[i][j][k] + prtrb < 0: minp[i][j][k] = 0
                        else: minp[i][j][k] += prtrb
                        num_change+=1

        manpltd_inputs.append(minp)
    return manpltd_inputs


def add_white_noise(inputs, model_path, selected_class, lrpmethod='simple', relevance_percentile=90, noise_std_dev=50):

    model_name = model_path.split('/')[1]

    fname = 'experiments/%s_%s_%d_%d' %(model_name, lrpmethod, selected_class, relevance_percentile)

    try:
        all_relevant_pixels = load_relevant_pixels(fname)
    except:
        all_relevant_pixels = find_relevant_pixels(inputs, model_path, lrpmethod, relevance_percentile)
        save_relevant_pixels(fname, all_relevant_pixels)
        print("NOT FOUND!")

    manpltd_inputs = []

    avg_num_relevant_pixels = 0
    for idx, inp in enumerate(inputs):
        minp = inp.copy()
        #minp = np.array([row[:] for row in inp])
        relevant_pixels = all_relevant_pixels[idx]
        for i, j, k in zip(relevant_pixels[0], relevant_pixels[1], relevant_pixels[2]):
            prtrb = np.random.normal(0,noise_std_dev)
            if minp[i][j][k] + prtrb > 255: minp[i][j][k] = 255 #maxed out
            elif minp[i][j][k] + prtrb < 0: minp[i][j][k] = 0
            else: minp[i][j][k] += prtrb

        manpltd_inputs.append(minp)


    return manpltd_inputs



def add_wn_random(inputs, model_path, selected_class, lrpmethod='simple', noise_std_dev=50, relevance_percentile=90):

    model_name = model_path.split('/')[1]

    fname = 'experiments/%s_%s_%d_%d' %(model_name, lrpmethod, selected_class, relevance_percentile)

    try:
        all_relevant_pixels = load_relevant_pixels(fname)
    except:
        all_relevant_pixels = find_relevant_pixels(inputs, model_path, lrpmethod, relevance_percentile)
        save_relevant_pixels(fname, all_relevant_pixels)
        print("NOT FOUND!")

    manpltd_inputs = []

    for idx, inp in enumerate(inputs):

        relevant_pixels = all_relevant_pixels[idx]
        relevant_pixels_merged = [(i,j,k) for i,j,k in zip(relevant_pixels[0], relevant_pixels[1], relevant_pixels[2])]

        data = [elem for elem in itertools.product(*[range(100),range(100),range(3)])]
#        data = [(i,j,k) for i in range(100) for j in range(100) for k in range(3)]
        data = list(set(data)-set(relevant_pixels_merged))

        random_pixels = sample(data, len(relevant_pixels[0]))

        minp = inp.copy()
        #minp = np.array([row[:] for row in inp])
        for [i,j,k] in random_pixels: #zip(random_pixels[0], random_pixels[1]):
            prtrb = np.random.normal(0,noise_std_dev)
            if minp[i][j][k] + prtrb > 255: minp[i][j][k] = 255 #maxed out
            elif minp[i][j][k] + prtrb < 0: minp[i][j][k] = 0
            else: minp[i][j][k] += prtrb

        manpltd_inputs.append(minp)

    return manpltd_inputs


