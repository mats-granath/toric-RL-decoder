import numpy as np
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################

# common system sizes are 3,5,7 and 9 
# grid size must be odd! 
system_size = 7

# valid network names: 
#   NN_11
#   NN_17
#   ResNet18
#   ResNet34
#   ResNet50
#   ResNet101
#   ResNet152
network = NN_17

# this file is stored in the network folder and contains the trained agent.  
NETWORK_FILE_NAME = 'Size_7_NN_17'

num_of_predictions = 1

# initialize RL class
rl = RL(Network=network,
        Network_name=NETWORK_FILE_NAME,
        system_size=system_size,
        device=device)

# initial syndrome error generation 
# generate syndrome with error probability 0.1 
prediction_list_p_error = [0.1]
# generate syndrome with a fixed amount of errors 
minimum_nbr_of_qubit_errors = int(system_size/2)+1 # minimum number of erorrs for logical qubit flip

# Generate folder structure, all results are stored in the data folder 
timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
PATH = 'data/prediction__' +str(NETWORK_FILE_NAME) +'__'+  timestamp
if not os.path.exists(PATH):
    os.makedirs(PATH)

# Path for the network to use for the prediction
PATH2 = 'network/'+str(NETWORK_FILE_NAME)+'.pt'
print('Prediction')
error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, failed_syndroms, ground_state_list, prediction_list_p_error, failure_rate = rl.prediction(
    num_of_predictions=num_of_predictions, 
    num_of_steps=75, 
    PATH=PATH2, 
    prediction_list_p_error=prediction_list_p_error,
    minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
    plot_one_episode=False)

# runtime of prediction
runtime = time.time()-start
runtime = runtime / 3600

print(error_corrected_list, 'error corrected')
print(ground_state_list, 'ground state conserved')
print(average_number_of_steps_list, 'average number of steps')
print(mean_q_list, 'mean q value')
print(runtime, 'h runtime')
  
# save training settings in txt file 
data_all = np.array([[NETWORK_FILE_NAME, failure_rate, num_of_predictions, error_corrected_list[0], ground_state_list[0],average_number_of_steps_list[0], mean_q_list[0], len(failed_syndroms)/2, runtime]])
np.savetxt(PATH + '/data_all.txt', data_all, header='network, failure_rate, error corrected, ground state conserved, average number of steps, mean q value, number of failed syndroms, runtime (h)', delimiter=',', fmt="%s")
