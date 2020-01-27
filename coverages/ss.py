from utils import get_layer_outs_new, get_layer_outs, percent_str
from collections import defaultdict


def default_sign_fn(x, _=None):
    """
    Default sign function implementation
    :param x:
    :param _: Ignored (included for interface consistency)
    :return: Sign value (+1 or -1)
    """
    return +1 if (x > 0).any() else -1


class SSCover:
    """
    Implements SS (Sign-Sign) coverage metric from "Testing Deep Neural Networks" by Sun et al.

    Class also supports incremental measurements using which one can observe the effect of new inputs to the coverage
    values.
    """

    def __init__(self, model, sign_fn=default_sign_fn, skip_layers=None):
        """
        :param model: Model
        :param sign_fn: Sign function that returns either +1 or -1 for a tensor
        :param skip_layers: Layers to be skipped (e.g. flatten layers)
        """
        self.cover_set = set()
        self.sign_sets = defaultdict(set)

        self.model = model
        self.sign_fn = sign_fn
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        self.layers = [model.layers[i] for i in range(len(model.layers)) if i not in skip_layers]

    def get_measure_state(self):
        return [self.cover_set, self.sign_sets]

    def set_measure_state(self, state):
        self.cover_set = state[0]
        self.sign_sets = state[1]

    def test(self, test_inputs):
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        total_pairs = 0

        for layer_index in range(len(outs) - 1):
            lower_layer_outs, upper_layer_outs = outs[layer_index], outs[layer_index + 1]
            lower_layer_fn, upper_layer_fn = self.layers[layer_index], self.layers[layer_index + 1]

            for lower_neuron_index in range(lower_layer_outs.shape[-1]):
                for upper_neuron_index in range(upper_layer_outs.shape[-1]):
                    total_pairs += 1

                    sign_set = self.sign_sets[(lower_neuron_index, upper_neuron_index)]
                    for input_index in range(len(test_inputs)):
                        rest_signs = \
                            tuple(self.sign_fn(lower_layer_outs[input_index, ..., i], lower_layer_fn)
                                  for i in range(lower_layer_outs.shape[-1]) if i != lower_neuron_index)

                        all_signs = (
                            self.sign_fn(lower_layer_outs[input_index, ..., lower_neuron_index], lower_layer_fn),
                            self.sign_fn(upper_layer_outs[input_index, ..., upper_neuron_index], upper_layer_fn),
                            rest_signs)

                        if (-1 * all_signs[0], -1 * all_signs[1], all_signs[2]) in sign_set:
                            self.cover_set.add(
                                ((layer_index, lower_neuron_index), (layer_index + 1, upper_neuron_index)))

                        sign_set.add(all_signs)

        covered_pair_count = len(self.cover_set)

        return percent_str(covered_pair_count, total_pairs), covered_pair_count, total_pairs, self.cover_set, outs


# Reference implementation
def _measure_ss_cover_naive(model, test_inputs, sign_fn=default_sign_fn, skip_layers=None):
    if skip_layers is None:
        skip_layers = []

    cover_set = set()
    outs = get_layer_outs(model, test_inputs, skip_layers)
    total_pairs = 0

    for layer_index in range(len(outs) - 1):
        lower_layer_outs, upper_layer_outs = outs[layer_index][0], outs[layer_index + 1][0]
        lower_layer_fn, upper_layer_fn = model.layers[layer_index], model.layers[layer_index + 1]

        print(layer_index)

        for lower_neuron_index in range(lower_layer_outs.shape[-1]):
            for upper_neuron_index in range(upper_layer_outs.shape[-1]):
                total_pairs += 1

                for input_index_i in range(len(test_inputs)):
                    for input_index_j in range(input_index_i + 1, len(test_inputs)):

                        covered = (sign_fn(lower_layer_outs[input_index_i][lower_neuron_index], lower_layer_fn) !=
                                   sign_fn(lower_layer_outs[input_index_j][lower_neuron_index], lower_layer_fn))

                        covered = covered and \
                                  (sign_fn(upper_layer_outs[input_index_i][upper_neuron_index], upper_layer_fn) !=
                                   sign_fn(upper_layer_outs[input_index_j][upper_neuron_index], upper_layer_fn))

                        for other_lower_neuron_index in range(lower_layer_outs.shape[-1]):
                            if other_lower_neuron_index == lower_neuron_index:
                                continue

                            covered = covered and \
                                      (sign_fn(lower_layer_outs[input_index_i][other_lower_neuron_index],
                                               lower_layer_fn) ==
                                       sign_fn(lower_layer_outs[input_index_j][other_lower_neuron_index],
                                               lower_layer_fn))

                        if covered:
                            cover_set.add(((layer_index, lower_neuron_index), (layer_index + 1, upper_neuron_index)))

    covered_pair_count = len(cover_set)

    return covered_pair_count, total_pairs, cover_set
