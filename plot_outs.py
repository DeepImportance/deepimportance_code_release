
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model, model_from_json
from utils import load_MNIST
from utils import get_layer_outs_new
from utils import filter_val_set
from utils import load_layerwise_relevances
from lrp_toolbox.model_io import read

experiment_folder = 'experiments'
selected_class = 0

X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
img_rows, img_cols = 28, 28

X_test, Y_test = filter_val_set(selected_class, X_test, Y_test)

relevant_neurons = load_layerwise_relevances('%s/%s_%d_%d_%d'
                                                %(experiment_folder,
                                                'LeNet5', 8, #rn
                                                selected_class, 7)) #layer
print(relevant_neurons)

json_file = open('neural_networks/LeNet5.json', 'r') #Read Keras model parameters (stored in JSON file)
file_content = json_file.read()
json_file.close()

model = model_from_json(file_content)
model.load_weights('neural_networks/LeNet5.h5')

# Compile the model before using
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

outfile = open('outfile.csv', 'w')
relfile = open('relfile.csv', 'w')



outs = get_layer_outs_new(model, X_test)
preds = model.predict(X_test)

lrpmodel = read('neural_networks/LeNet5.txt', 'txt')  # 99.16% prediction accuracy
lrpmodel.drop_softmax_output_layer()  # drop softnax output layer for analysis

Rs = []
for inp in X_test:
    ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))

    mask = np.zeros_like(ypred)
    mask[:,np.argmax(ypred)] = 1
    Rinit = ypred*mask

    R_inp, R_all = lrpmodel.lrp(Rinit,'alphabeta',3)
    Rs.append(R_all[-1])


for i in range(len(X_test)): #100 inputs
    #out_data = []
    #rel_data = []
    out_row = ''
    rel_row = ''
    for j in range(outs[-3].shape[-1]):
        #out_data.append(outs[-3][j][i])
        #rel_data.append(Rs[j][0][i])
        out_row += str(outs[-3][i][j]) + ','
        rel_row += str(Rs[i][0][j]) + ','
    out_row += str(Y_test[i].argmax(axis=-1)) + ',' + str(preds[i].argmax(axis=-1)) + '\n'
    rel_row += str(Y_test[i].argmax(axis=-1)) + ',' + str(preds[i].argmax(axis=-1)) + '\n'
    outfile.write(out_row)
    relfile.write(rel_row)


outfile.close()
relfile.close()

#    plt.clf()
#    plt.plot(range(10), out_data)
#    plt.plot(range(10), rel_data)
#    plt.savefig("./plots/plt"+str(i)+".png")

'''
for i in range(outs[-3].shape[-1]):
    out_data = []
    for j in range(10): #100 inputs
        out_data.append(Rs[j][0][i])

    plt.clf()
    plt.plot(range(10), out_data)
    plt.savefig("./plots/rel"+str(i)+".png")

'''
