import os
import datetime

from utils import get_python_version

python_version = get_python_version()

attacks       = ['bim']#, 'jsma', 'cw', 'bim']
labels        = [0,1,2,3,4,5,6,7,8,9]
qq            = [2,3]
rel_num       = [6,8,10, 12]
model_names   = ['neural_networks/cifar_original']
#model_names   = ['neural_networks/LeNet1', 'neural_networks/LeNet4', 'neural_networks/LeNet5']# 'neural_networks/cifar40_128']
approaches    = ['lsa', 'dsa']

for seed in range(1,6):
    for model in model_names:
#    for app in approaches:
        #for l in labels:
            #for rn in rel_num:
        command = 'python validation_cifar.py -DS cifar10 -M %s -S %d -A lsa' %(model, seed)
        os.system(command)

exit()
for r in rn:
    for model in model_names:
        for label in labels:
            for q in qq:
                for at in attacks:
                    command = 'python coverage_runner.py -A cc -M %s -DS cifar10 -C %d \
                                            -Q %d -RN %d -ADV %s' %(model, label,\
                                                                    q, r, at)
                    os.system(command)

