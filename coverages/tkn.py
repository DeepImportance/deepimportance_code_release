# -*- coding: utf-8 -*-
import numpy as np
from utils import get_layer_outs, get_layer_outs_new, percent_str


class DeepGaugeLayerLevelCoverage:
    """
    Implements TKN and TKN-with-pattern coverage metrics from "DeepGauge: Multi-Granularity Testing Criteria for Deep
    Learning Systems" by Ma et al.

    Supports incremental measurements using which one can observe the effect of new inputs to the coverage
    values.
    """

    def __init__(self, model, k, skip_layers=None):
        """
        :param model: Model
        :param k: k parameter (see the paper)
        :param skip_layers: Layers to be skipped (e.g. flatten layers)
        """
        self.activation_table = {}
        self.pattern_set = set()

        self.model = model
        self.k = k
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)

    def get_measure_state(self):
        return [self.activation_table, self.pattern_set]

    def set_measure_state(self, state):
        self.activation_table = state[0]
        self.pattern_set = state[1]

    def test(self, test_inputs):
        """
        :param test_inputs: Inputs
        :return: Tuple consisting of coverage results along with the measurements that are used to compute the
        coverages. 0th element is the TKN value and 3th element is the pattern count for TKN-with-pattern.
        """
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        neuron_count_by_layer = {}

        layer_count = len(outs)

        inc_cnt_tkn = 0
        for input_index in range(len(test_inputs)):  # out_for_input is output of layer for single input
            pattern = []

            inc_flag = False
            for layer_index in range(layer_count):  # layer_out is output of layer for all inputs
                out_for_input = outs[layer_index][input_index]

                neuron_outs = np.zeros((out_for_input.shape[-1],))
                neuron_count_by_layer[layer_index] = len(neuron_outs)
                for i in range(out_for_input.shape[-1]):
                    neuron_outs[i] = np.mean(out_for_input[..., i])

                top_k_neuron_indexes = (np.argsort(neuron_outs, axis=None)[-self.k:len(neuron_outs)])
                pattern.append(tuple(top_k_neuron_indexes))

                for neuron_index in top_k_neuron_indexes:
                    if not (layer_index, neuron_index) in self.activation_table: inc_flag = True
                    self.activation_table[(layer_index, neuron_index)] = True

                if layer_index + 1 == layer_count:
                    self.pattern_set.add(tuple(pattern))

            if inc_flag:
                inc_cnt_tkn += 1

        neuron_count = sum(neuron_count_by_layer.values())
        covered = len(self.activation_table.keys())

        print(percent_str(covered, neuron_count))
        # TKNC                                                         #TKNP
        return percent_str(covered, neuron_count), covered, neuron_count, len(self.pattern_set), outs, inc_cnt_tkn
