import numpy as np
from sklearn import cluster

from utils import save_quantization, load_quantization, save_totalR, load_totalR
from utils import save_layerwise_relevances, load_layerwise_relevances
from utils import get_layer_outs_new
from lrp_toolbox.model_io import write, read

from sklearn.metrics import silhouette_score

experiment_folder = 'experiments'
model_folder      = 'neural_networks'

class ImportanceDrivenCoverage:
    def __init__(self,model, model_name, num_relevant_neurons, selected_class, subject_layer,
                 train_inputs, train_labels):
        self.covered_combinations = ()

        self.model = model
        self.model_name = model_name
        self.num_relevant_neurons = num_relevant_neurons
        self.selected_class = selected_class
        self.subject_layer = subject_layer
        self.train_inputs = train_inputs
        self.train_labels = train_labels


    def get_measure_state(self):
        return self.covered_combinations

    def set_measure_state(self, covered_combinations):
        self.covered_combinations = covered_combinations

    def test(self, test_inputs):
        #########################
        #1.Find Relevant Neurons#
        #########################

        try:
            relevant_neurons = load_layerwise_relevances('%s/%s_%d_%d_%d'
                                                         %(experiment_folder,
                                                           self.model_name,
                                                           self.num_relevant_neurons,
                                                           self.selected_class,
                                                           self.subject_layer))
        except:
            print("RN NOT FOUND!")
            # Convert keras model into txt
            model_path = model_folder + '/' + self.model_name
            write(model_path, model_path, num_channels=test_inputs[0].shape[-1], fmt='keras_txt')

            lrpmodel = read(model_path + '.txt', 'txt')  # 99.16% prediction accuracy
            lrpmodel.drop_softmax_output_layer()  # drop softnax output layer for analysis

            relevant_neurons, least_relevant_neurons, total_R = find_relevant_neurons(
                self.model, lrpmodel, self.train_inputs, self.train_labels,
                self.subject_layer, self.num_relevant_neurons, None, 'sum')

            save_totalR(total_R, '%s/%s_%s_%d'
                        %(experiment_folder, self.model_name,
                          'totalR', self.selected_class), 0)


        ####################################
        #2.Quantize Relevant Neuron Outputs#
        ####################################
        if 'conv' in self.model.layers[self.subject_layer].name: is_conv = True
        else: is_conv = False

        try:
            qtized = load_quantization('%s/%s_%d_%d_%d_silhouette'
                                %(experiment_folder,
                                self.model_name,
                                self.selected_class,
                                self.subject_layer,
                                self.num_relevant_neurons),0)
        except:
            print("Q NOT FOUND!")
            train_layer_outs = get_layer_outs_new(self.model, np.array(self.train_inputs))

            qtized = quantizeSilhouette(train_layer_outs[self.subject_layer], is_conv,
                              relevant_neurons)
            save_quantization(qtized, '%s/%s_%d_%d_%d_silhouette'
                              %(experiment_folder,
                                self.model_name,
                                self.selected_class,
                                self.subject_layer,
                                self.num_relevant_neurons),0)


        ####################
        #3.Measure coverage#
        ####################
        test_layer_outs = get_layer_outs_new(self.model, np.array(test_inputs))

        coverage, covered_combinations = measure_idc(self.model, self.model_name,
                                                                test_inputs, self.subject_layer,
                                                                relevant_neurons,
                                                                self.selected_class,
                                                                test_layer_outs, qtized, is_conv,
                                                                self.covered_combinations)

        return coverage, covered_combinations#,# len(qtized[0])


def quantize(out_vectors, conv, relevant_neurons, n_clusters=3):
    #if conv: n_clusters+=1
    quantized_ = []

    for i in range(out_vectors.shape[-1]):
        out_i = []
        for l in out_vectors:
            if conv: #conv layer
                out_i.append(np.mean(l[...,i]))
            else:
                out_i.append(l[i])

        #If it is a convolutional layer no need for 0 output check
        if not conv: out_i = filter(lambda elem: elem != 0, out_i)
        values = []
        if not len(out_i) < 10: #10 is threshold of number positives in all test input activations
            kmeans = cluster.KMeans(n_clusters=n_clusters)
            kmeans.fit(np.array(out_i).reshape(-1, 1))
            values = kmeans.cluster_centers_.squeeze()
        values = list(values)
        values = limit_precision(values)

        #if not conv: values.append(0) #If it is convolutional layer we dont add  directly since thake average of whole filter.

        quantized_.append(values)

    quantized_ = [quantized_[rn] for rn in relevant_neurons]

    return quantized_


def quantizeSilhouette(out_vectors, conv, relevant_neurons):
    #if conv: n_clusters+=1
    quantized_ = []

    for i in range(out_vectors.shape[-1]):
        if i not in relevant_neurons: continue

        out_i = []
        for l in out_vectors:
            if conv: #conv layer
                out_i.append(np.mean(l[...,i]))
            else:
                out_i.append(l[i])

        #If it is a convolutional layer no need for 0 output check
        #if not conv: out_i = [item for item in out_i if item != 0]
        out_i = filter(lambda elem: elem != 0, out_i)
        values = []

        if not len(out_i) < 10: #10 is threshold of number positives in all test input activations

            clusterSize = range(2, 5)#[2, 3, 4, 5]
            clustersDict = {}
            for clusterNum in clusterSize:
                kmeans          = cluster.KMeans(n_clusters=clusterNum)
                clusterLabels   = kmeans.fit_predict(np.array(out_i).reshape(-1, 1))
                silhouetteAvg   = silhouette_score(np.array(out_i).reshape(-1, 1), clusterLabels)
                clustersDict [silhouetteAvg] = kmeans

            maxSilhouetteScore = max(clustersDict.keys())
            bestKMean          = clustersDict[maxSilhouetteScore]

            values = bestKMean.cluster_centers_.squeeze()
        values = list(values)
        values = limit_precision(values)

        #if not conv: values.append(0) #If it is convolutional layer we dont add  directly since thake average of whole filter.
        if len(values) == 0: values.append(0)

        quantized_.append(values)
    #quantized_ = [quantized_[rn] for rn in relevant_neurons]

    return quantized_

def limit_precision(values, prec=2):
    limited_values = []
    for v in values:
        limited_values.append(round(v,prec))

    return limited_values


def determine_quantized_cover(lout, quantized):
    covered_comb = []
    for idx, l in enumerate(lout):
        #if l == 0:
        #    covered_comb.append(0)
        #else:
        closest_q = min(quantized[idx], key=lambda x:abs(x-l))
        covered_comb.append(closest_q)

    return covered_comb


def measure_idc(model, model_name, test_inputs, subject_layer,
                                   relevant_neurons, sel_class,
                                   test_layer_outs, qtized, is_conv,
                                   covered_combinations=()):

    subject_layer = subject_layer - 1
    for test_idx in range(len(test_inputs)):
        if is_conv:
            lout = []
            for r in relevant_neurons:
                lout.append(np.mean(test_layer_outs[subject_layer][test_idx][...,r]))
        else:
            lout = test_layer_outs[subject_layer][test_idx][relevant_neurons]

        comb_to_add = determine_quantized_cover(lout, qtized)

        if comb_to_add not in covered_combinations:
            covered_combinations += (comb_to_add,)

    max_comb = 1#q_granularity**len(relevant_neurons)
    for q in qtized:
        max_comb *= len(q)

    covered_num = len(covered_combinations)
    coverage = float(covered_num)/max_comb

    return coverage*100, covered_combinations


def find_relevant_neurons(kerasmodel, lrpmodel, inps, outs, subject_layer, \
            num_rel, lrpmethod=None, final_relevance_method='sum'):

    final_relevants = np.zeros([1, kerasmodel.layers[subject_layer].output_shape[-1]])

    totalR = None
    cnt = 0
    for inp in inps:
        cnt+=1
        ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))

        #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
        mask = np.zeros_like(ypred)
        mask[:,np.argmax(ypred)] = 1
        Rinit = ypred*mask

        if not lrpmethod:
            R_inp, R_all = lrpmodel.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'epsilon':
            R_inp, R_all = lrpmodel.lrp(Rinit,'epsilon',0.01)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'alphabeta':
            R_inp, R_all = lrpmodel.lrp(Rinit,'alphabeta',3)     #as Eq(60) from DOI: 10.1371/journal.pone.0130140
        else:
            print('Unknown LRP method!')
            raise Exception

        if totalR:
            for idx, elem in enumerate(totalR):
                totalR[idx] = elem + R_all[idx]

        else: totalR = R_all

    #      THE MOST RELEVANT                               THE LEAST RELEVANT
    return np.argsort(final_relevants)[0][::-1][:num_rel], np.argsort(final_relevants)[0][:num_rel], totalR



