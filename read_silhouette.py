from utils import load_quantization

experiment_folder = "experiments"
model_name = "LeNet5"
selected_class = 0
num_relevant_neurons = 12
subject_layer = 7

ff = open('num_clusters_lenet5.log', 'w')

for i in range(10):
    qtized = load_quantization('%s/%s_%d_%d_%d_silhouette'
                                    %(experiment_folder,
                                    model_name,
                                    i,
                                    subject_layer,
                                    num_relevant_neurons),0)

    row = 'Class ' + str(i) + ': '
    lens = []
    for q in qtized:
        row += str(len(q)) + ' '
        lens.append(len(q))

    row += str(reduce(lambda x, y: x*y, lens[:6])) + ' '
    row += str(reduce(lambda x, y: x*y, lens[:8])) + ' '
    row += str(reduce(lambda x, y: x*y, lens[:10])) + ' '
    row += str(reduce(lambda x, y: x*y, lens[:12])) + ' '

    row += '\n'
    ff.write(row)

ff.close()

