import numpy as np
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code

from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

##########################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NETWORK = NN_17
#NETWORK = ResNet18
SYSTEM_SIZE = 7

continue_training = False
NETWORK_FILE_NAME = 'size_7_NN_17'

rl = RL(Network=NETWORK,
        Network_name=NETWORK_FILE_NAME,
        system_size=SYSTEM_SIZE,
        p_error=0.1,
        replay_memory_capacity=20000, 
        learning_rate=0.00025,
        discount_factor=0.95,
        max_nbr_actions_per_episode=5,
        device=device,
        replay_memory='proportional')   # proportional  
                                        # uniform


# generate folder structure 
timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
PATH = 'data/training__' +str(NETWORK_FILE_NAME) +'_'+str(SYSTEM_SIZE)+'__' + timestamp
PATH_epoch = PATH + '/network_epoch'
if not os.path.exists(PATH):
    os.makedirs(PATH)
    os.makedirs(PATH_epoch)


if continue_training == True:
    print('continue training')
    PATH2 = 'network/'+str(NETWORK_FILE_NAME)+'.pt'
    rl.load_network(PATH2)
                                 
rl.train_for_n_epochs(training_steps=50,
                    num_of_predictions=1,
                    epochs=1,
                    target_update=10,
                    reward_definition=4,
                    optimizer='Adam',
                    batch_size=1,
                    directory_path = PATH,
                    prediction_list_p_error=[0.1],
                    replay_start_size=48)  
                                                       


""" rl.train_for_n_epochs(training_steps=10000,
                            num_of_predictions=100,
                            epochs=100,
                            target_update=1000,
                            reward_definition=4,
                            optimizer='Adam',
                            batch_size=32,
                            directory_path = PATH,
                            prediction_list_p_error=[0.1],
                            nbr_of_qubit_errors=0)   """
               
