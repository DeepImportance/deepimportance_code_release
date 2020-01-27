from utils import load_data, save_data
import numpy as np

experiment_folder = 'experiments'
model_names = ['LeNet1', 'LeNet4', 'LeNet5']
adv_types = ['fgsm','jsma', 'bim']#, 'cw' ]
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for model_name in model_names:
    for adv_type in adv_types:
        X_adv=np.array([])
        for selected_class in classes:
            adv = load_data('%s/%s_%d_%s_adversarial' %(experiment_folder, model_name,
                                                        selected_class, adv_type))
            if X_adv.shape[0] > 0: X_adv = np.concatenate((X_adv,adv),axis=0)
            else: X_adv = adv

        save_data(X_adv, '%s/%s_-1_%s_adversarial' %(experiment_folder, model_name,
                                             adv_type))
