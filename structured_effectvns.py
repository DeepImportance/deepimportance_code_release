import os
import datetime

from utils import get_python_version

python_version = get_python_version()

#model_names   = ['neural_networks/cifar_original']#,
model_names   = ['neural_networks/LeNet1', 'neural_networks/LeNet4', 'neural_networks/LeNet5']# 'neural_networks/cifar40_128']
labels        = [0,1,2,3,4,5,6,7,8,9]
rel_num       = [6,8]
q_granularity = [3, 4]
adv_types     = ['cw'] #'gauss', 'jsma', 'fgsm', 'bim']#, 'cw', 'gauss']
approaches    = ['lsa','dsa'] #'nc', 'kmnc', 'tknc', 'lsa', 'dsa']

for model in model_names:
    for adv in adv_types:
        for app in approaches:
            command = str("python effectiveness.py -M %s -DS mnist -A %s -ADV %s -C -1" %(model, app, adv))
            os.system(command)
            print(datetime.datetime.now())

