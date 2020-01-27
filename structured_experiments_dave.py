import os
import datetime

from utils import get_python_version

python_version = get_python_version()

model_names   = ['neural_networks/dave2']
labels        = [0,1,2,3,4,5,6,7,8,9]
rel_num       = [6,8,10,12]
q_granularity = [3, 4]
adv_types     = ['gauss', 'jsma', 'fgsm', 'bim']#, 'cw', 'gauss']
approaches    = ['cc'] #'nc', 'kmnc', 'tknc', 'lsa', 'dsa']

for seed in range(1,6):
    for model in model_names:
#        for rn in rel_num:
        command = str("python validation_dave.py -M %s -A lsa -S %d -DS drive" %(model, seed))
        os.system(command)
        print(datetime.datetime.now())


