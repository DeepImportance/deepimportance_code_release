import sys

sys.path.append('../')

import numpy as np
from utils import get_layer_outs_new, percent_str
from collections import defaultdict


def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin

    return X_scaled


class NeuronCoverage:
    """
    Implements Neuron Coverage metric from "DeepXplore: Automated Whitebox Testing of Deep Learning Systems" by Pei
    et al.

    Supports incremental measurements using which one can observe the effect of new inputs to the coverage
    values.
    """

    def __init__(self, model, scaler=default_scale, threshold=0, skip_layers=None):
        self.activation_table = defaultdict(bool)

        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)

    def get_measure_state(self):
        return [self.activation_table]

    def set_measure_state(self, state):
        self.activation_table = state[0]

    def test(self, test_inputs):
        """
        :param test_inputs: Inputs
        :return: Tuple containing the coverage and the measurements used to compute the coverage. 0th element is the
        percentage neuron coverage value.
        """
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
        used_inps = []
        nc_cnt = 0
        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            inp_cnt = 0
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    if not self.activation_table[(layer_index, neuron_index)] and np.mean(
                            out_for_input[..., neuron_index]) > self.threshold and inp_cnt not in used_inps:
                        used_inps.append(inp_cnt)
                        nc_cnt += 1
                    self.activation_table[(layer_index, neuron_index)] = self.activation_table[
                                                                             (layer_index, neuron_index)] or np.mean(
                        out_for_input[..., neuron_index]) > self.threshold

                inp_cnt += 1

        covered = len([1 for c in self.activation_table.values() if c])
        total = len(self.activation_table.keys())

        return percent_str(covered, total), covered, total, outs, nc_cnt
