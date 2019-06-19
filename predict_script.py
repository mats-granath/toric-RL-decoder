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

# CHANGE NETWORK, SYSTEMSIZE

system_size = 7
#network = ResNet18
network = NN_17

rl = RL(Network=network,
        system_size=system_size,
        device=device)
#######################################################################################################
#NETWORK = 'ResNet18_5'
NETWORK = 'size_7_NN_17'

prediction_list_p_error = [0.1]
num_of_predictions = 1

print('Prediction')
nbr_of_qubit_errors = int(system_size/2)+2

timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
PATH = 'data/prediction__' +str(NETWORK) +'__'+  timestamp
if not os.path.exists(PATH):
    os.makedirs(PATH)

PATH2 = 'network/'+str(NETWORK)+'.pt'
error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, failed_syndroms, ground_state_list, prediction_list_p_error, failure_rate = rl.prediction(
    num_of_predictions=num_of_predictions, 
    num_of_steps=75, 
    PATH=PATH2, 
    prediction_list_p_error=prediction_list_p_error,
    nbr_of_qubit_errors=nbr_of_qubit_errors,
    directory_path=PATH,
    plot_one_episode=False)

print(error_corrected_list, 'error corrected')
print(ground_state_list, 'ground state conserved')
print(average_number_of_steps_list, 'average number of steps')
print(mean_q_list, 'mean q value')

runtime = time.time()-start
runtime = runtime / 3600
blubb = time.strftime('%H, %M, %S')

print(runtime, 'h runtime')

data_all = np.array([[NETWORK, failure_rate, num_of_predictions, error_corrected_list[0], ground_state_list[0],average_number_of_steps_list[0], mean_q_list[0], len(failed_syndroms)/2, runtime]])
  
# save training settings in txt file 
np.savetxt(PATH + '/data_all.txt', data_all, header='network, failure_rate, error corrected, ground state conserved, average number of steps, mean q value, number of failed syndroms, runtime (h)', delimiter=',', fmt="%s")

