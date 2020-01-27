import os
import datetime

from utils import get_python_version

python_version = get_python_version()

#model_names   = ['neural_networks/cifar_original']#,
model_names   = ['neural_networks/LeNet1', 'neural_networks/LeNet4', 'neural_networks/LeNet5']# 'neural_networks/cifar40_128']
labels        = [0,1,2,3,4,5,6,7,8,9]
rel_num       = [6,8,10,12]
q_granularity = [3, 4]
adv_types     = ['gauss', 'jsma', 'fgsm', 'bim']#, 'cw', 'gauss']
approaches    = ['lsa','dsa'] #'nc', 'kmnc', 'tknc', 'lsa', 'dsa']

for seed in range(1,6):
    for model in model_names:
        #for app in approaches:
        #for l in labels:
        #    for q in q_granularity:
            #for rn in rel_num:
            #if (rn == 6 and 'cifar' in model) and (rn == 10 and 'LeNet' in model): continue
        command = str("python validation_mnist.py -M %s -DS mnist -A tknc -S %d" %(model, seed))
        os.system(command)
        print(datetime.datetime.now())

exit()


attacks       = ['cw', 'bim']
labels        = [0,1,2,3,4,5,6,7,8,9]
qq            = [2]
rn            = [8]
model_names   = ['neural_networks/cifar40_128']

for r in rn:
    for model in model_names:
        for label in labels:
            for q in qq:
                for at in attacks:
                    command = 'python coverage_runner.py -A tknc -M %s -DS cifar10 -C %d \
                                            -Q %d -RN %d -ADV %s' %(model, label,\
                                                                    q, r, at)
                    os.system(command)


exit()
