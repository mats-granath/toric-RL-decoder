# standard libraries
import numpy as np
import random
import time
from collections import namedtuple, Counter
import operator
import os
from copy import deepcopy
import heapq
# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
# import from other files
from .toric_model import Toric_code
from .toric_model import Action
from .toric_model import Perspective
from .Replay_memory import Replay_memory_uniform, Replay_memory_prioritized
# import networks 
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .util import incremental_mean, convert_from_np_to_tensor, Transition


class RL():
    def __init__(self, Network, Network_name, system_size=int, p_error=0.1, replay_memory_capacity=int, learning_rate=0.00025,
                discount_factor=0.95, number_of_actions=3, max_nbr_actions_per_episode=50, device='cpu', replay_memory='uniform'):
        # device
        self.device = device
        # Toric code
        if system_size%2 > 0:
            self.toric = Toric_code(system_size)
        else:
            raise ValueError('Invalid system_size, please use only odd system sizes.')
        self.grid_shift = int(system_size/2)
        self.max_nbr_actions_per_episode = max_nbr_actions_per_episode
        self.system_size = system_size
        self.p_error = p_error
        # Replay Memory
        self.replay_memory_capacity = replay_memory_capacity
        self.replay_memory = replay_memory
        if self.replay_memory == 'proportional':
            self.memory = Replay_memory_prioritized(replay_memory_capacity, 0.6) # alpha
        elif self.replay_memory == 'uniform':
            self.memory = Replay_memory_uniform(replay_memory_capacity)
        else:
            raise ValueError('Invalid memory type, please use only proportional or uniform.')
        # Network
        self.network_name = Network_name
        self.network = Network
        if Network == ResNet18 or Network == ResNet34 or Network == ResNet50 or Network == ResNet101 or Network == ResNet152:
            self.policy_net = self.network()
        else:
            self.policy_net = self.network(system_size, number_of_actions, device)
        self.target_net = deepcopy(self.policy_net)
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.learning_rate = learning_rate
        # hyperparameters RL
        self.discount_factor = discount_factor
        self.number_of_actions = number_of_actions


    def save_network(self, PATH):
        torch.save(self.policy_net, PATH)


    def load_network(self, PATH):
        self.policy_net = torch.load(PATH, map_location='cpu')
        self.target_net = deepcopy(self.policy_net)
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)


    def experience_replay(self, criterion, optimizer, batch_size):
        self.policy_net.train()
        self.target_net.eval()
        # get transitions and unpack them to minibatch
        transitions, weights, indices = self.memory.sample(batch_size, 0.4) # beta parameter 
        mini_batch = Transition(*zip(*transitions))
        # unpack action batch
        batch_actions = Action(*zip(*mini_batch.action))
        batch_actions = np.array(batch_actions.action) - 1
        batch_actions = torch.Tensor(batch_actions).long()
        batch_actions = batch_actions.to(self.device)
        # preprocess batch_input and batch_target_input for the network
        batch_state = self.get_batch_input(mini_batch.state)
        batch_next_state = self.get_batch_input(mini_batch.next_state)
        # preprocess batch_terminal and batch reward
        batch_terminal = convert_from_np_to_tensor(np.array(mini_batch.terminal)) 
        batch_terminal = batch_terminal.to(self.device)
        batch_reward = convert_from_np_to_tensor(np.array(mini_batch.reward))
        batch_reward = batch_reward.to(self.device)
        # compute policy net output
        output = self.policy_net(batch_state)
        output = output.gather(1, batch_actions.view(-1, 1)).squeeze(1)    
        # compute target network output 
        target_output = self.get_target_network_output(batch_next_state, batch_size)
        target_output = target_output.to(self.device)
        y = batch_reward + (batch_terminal * self.discount_factor * target_output)
        # compute loss and update replay memory
        loss = self.get_loss(criterion, optimizer, y, output, weights, indices)
        # backpropagate loss
        loss.backward()
        optimizer.step()


    def get_loss(self, criterion, optimizer, y, output, weights, indices):
        loss = criterion(y, output)
        optimizer.zero_grad()
        # for prioritized experience replay
        if self.replay_memory == 'proportional':
            loss = convert_from_np_to_tensor(np.array(weights)) * loss.cpu()
            priorities = loss
            priorities = np.absolute(priorities.detach().numpy())
            self.memory.priority_update(indices, priorities)
        return loss.mean()


    def get_network_output_next_state(self, batch_next_state=float, batch_size=int, action_index=None):
        self.target_net.eval()
        self.policy_net.eval()
        # init matrices
        batch_network_output = np.zeros(batch_size)
        batch_perspectives = np.zeros(shape=(batch_size, 2, self.system_size, self.system_size))
        batch_actions = np.zeros(batch_size)
        for i in range(batch_size):
            if (batch_next_state[i].cpu().sum().item() == 0):
                batch_perspectives[i,:,:,:] = np.zeros(shape=(2, self.system_size, self.system_size))
            else:
                perspectives = self.toric.generate_perspective(self.grid_shift, batch_next_state[i].cpu())
                perspectives = Perspective(*zip(*perspectives))
                perspectives = np.array(perspectives.perspective)
                perspectives = convert_from_np_to_tensor(perspectives)
                perspectives = perspectives.to(self.device)
                # select greedy action 
                with torch.no_grad():        
                    net_output = self.target_net(perspectives)
                    q_values_table = np.array(net_output.cpu())
                    row, col = np.where(q_values_table == np.max(q_values_table))
                    if action_index[i] == None:
                        batch_network_output[i] = q_values_table[row[0], col[0]]                            
                    elif action_index[i] != None:
                        action_from_policy_net = int(action_index[i])
                        batch_network_output[i] = q_values_table[row[0], action_from_policy_net]
                    perspective = perspectives[row[0]]
                    perspective = np.array(perspective.cpu())
                    batch_perspectives[i,:,:,:] = perspective
                    batch_actions[i] = col[0]
                    batch_network_output[i] = q_values_table[row[0], col[0]]
        batch_network_output = convert_from_np_to_tensor(batch_network_output)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        return batch_network_output, batch_perspectives, batch_actions


    def get_target_network_output(self, batch_next_state, batch_size):
        with torch.no_grad():
            action_index = np.full(shape=(batch_size), fill_value=None)
            target_output,_,_ = self.get_network_output_next_state(batch_next_state=batch_next_state, 
                                                                        batch_size=batch_size, 
                                                                        action_index=action_index)
        return target_output


    def get_batch_input(self, state_batch):
        batch_input = np.stack(state_batch, axis=0)
        batch_input = convert_from_np_to_tensor(batch_input)
        return batch_input.to(self.device)


    def train(self, training_steps=int, target_update=int, epsilon_start=1.0, num_of_epsilon_steps=10, 
        epsilon_end=0.1, reach_final_epsilon=0.5, optimizer=str,
        batch_size=int, replay_start_size=int, minimum_nbr_of_qubit_errors=0):
        # set network to train mode
        self.policy_net.train()
        # define criterion and optimizer
        criterion = nn.MSELoss(reduction='none')
        if optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        elif optimizer == 'Adam':    
            optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # init counters
        steps_counter = 0
        update_counter = 1
        iteration = 0
        # define epsilon steps 
        epsilon = epsilon_start
        num_of_steps = np.round(training_steps/num_of_epsilon_steps)
        epsilon_decay = np.round((epsilon_start-epsilon_end)/num_of_epsilon_steps, 5)
        epsilon_update = num_of_steps * reach_final_epsilon
        # main loop over training steps 
        while iteration < training_steps:
            num_of_steps_per_episode = 0
            # initialize syndrom
            self.toric = Toric_code(self.system_size)
            terminal_state = 0
            # generate syndroms
            while terminal_state == 0:
                if minimum_nbr_of_qubit_errors == 0:
                    self.toric.generate_random_error(self.p_error)
                else:
                    self.toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
                terminal_state = self.toric.terminal_state(self.toric.current_state)
            # solve one episode
            while terminal_state == 1 and num_of_steps_per_episode < self.max_nbr_actions_per_episode and iteration < training_steps:
                num_of_steps_per_episode += 1
                num_of_epsilon_steps += 1
                steps_counter += 1
                iteration += 1
                # select action using epsilon greedy policy
                action = self.select_action(number_of_actions=self.number_of_actions,
                                            epsilon=epsilon, 
                                            grid_shift=self.grid_shift)
                self.toric.step(action)
                reward = self.get_reward()
                # generate memory entry
                perspective, action_memory, reward, next_perspective, terminal = self.toric.generate_memory_entry(
                    action, reward, self.grid_shift)    
                # save transition in memory
                self.memory.save(Transition(perspective, action_memory, reward, next_perspective, terminal), 10000) # max priority
            
                # experience replay
                if steps_counter > replay_start_size:
                    update_counter += 1
                    self.experience_replay(criterion,
                                            optimizer,
                                            batch_size)
                # set target_net to policy_net
                if update_counter % target_update == 0:
                    self.target_net = deepcopy(self.policy_net)
                # update epsilon
                if (update_counter % epsilon_update == 0):
                    epsilon = np.round(np.maximum(epsilon - epsilon_decay, epsilon_end), 3)
                # set next_state to new state and update terminal state
                self.toric.current_state = self.toric.next_state
                terminal_state = self.toric.terminal_state(self.toric.current_state)


    def get_reward(self):
        terminal = np.all(self.toric.next_state==0)
        if terminal == True:
            reward = 100
        else:
            defects_state = np.sum(self.toric.current_state)
            defects_next_state = np.sum(self.toric.next_state)
            reward = defects_state - defects_next_state

        return reward


    def select_action(self, number_of_actions=int, epsilon=float, grid_shift=int):
        # set network in evluation mode 
        self.policy_net.eval()
        # generate perspectives 
        perspectives = self.toric.generate_perspective(grid_shift, self.toric.current_state)
        number_of_perspectives = len(perspectives)
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        #choose action using epsilon greedy approach
        rand = random.random()
        if(1 - epsilon > rand):
            # select greedy action 
            with torch.no_grad():        
                policy_net_output = self.policy_net(batch_perspectives)
                q_values_table = np.array(policy_net_output.cpu())
                row, col = np.where(q_values_table == np.max(q_values_table))
                perspective = row[0]
                max_q_action = col[0] + 1
                step = Action(batch_position_actions[perspective], max_q_action)
        # select random action
        else:
            random_perspective = random.randint(0, number_of_perspectives-1)
            random_action = random.randint(1, number_of_actions)
            step = Action(batch_position_actions[random_perspective], random_action)  

        return step


    def select_action_prediction(self, number_of_actions=int, epsilon=float, grid_shift=int, prev_action=float):
        # set network in eval mode
        self.policy_net.eval()
        # generate perspectives
        perspectives = self.toric.generate_perspective(grid_shift, self.toric.current_state)
        number_of_perspectives = len(perspectives)
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        # generate action value for different perspectives 
        with torch.no_grad():
            policy_net_output = self.policy_net(batch_perspectives)
            q_values_table = np.array(policy_net_output.cpu())
        #choose action using epsilon greedy approach
        rand = random.random()
        if(1 - epsilon > rand):
            # select greedy action 
            row, col = np.where(q_values_table == np.max(q_values_table))
            perspective = row[0]
            max_q_action = col[0] + 1
            step = Action(batch_position_actions[perspective], max_q_action)
            if prev_action == step:
                res = heapq.nlargest(2, q_values_table.flatten())
                row, col = np.where(q_values_table == res[1])
                perspective = row[0]
                max_q_action = col[0] + 1
                step = Action(batch_position_actions[perspective], max_q_action)
            q_value = q_values_table[row[0], col[0]]
        # select random action
        else:
            random_perspective = random.randint(0, number_of_perspectives-1)
            random_action = random.randint(1, number_of_actions)
            q_value = q_values_table[random_perspective, random_action-1]
            step = Action(batch_position_actions[random_perspective], random_action)

        return step, q_value


    def prediction(self, num_of_predictions=1, epsilon=0.0, num_of_steps=50, PATH=None, plot_one_episode=False, 
        show_network=False, show_plot=False, prediction_list_p_error=float, minimum_nbr_of_qubit_errors=0, print_Q_values=False, save_prediction=True):
        # load network for prediction and set eval mode 
        if PATH != None:
            self.load_network(PATH)
        self.policy_net.eval()
        # init matrices 
        ground_state_list = np.zeros(len(prediction_list_p_error))
        error_corrected_list = np.zeros(len(prediction_list_p_error))
        average_number_of_steps_list = np.zeros(len(prediction_list_p_error))
        mean_q_list = np.zeros(len(prediction_list_p_error))
        failed_syndroms = []
        failure_rate = 0
        # loop through different p_error
        for i, p_error in enumerate(prediction_list_p_error):
            ground_state = np.ones(num_of_predictions, dtype=bool)
            error_corrected = np.zeros(num_of_predictions)
            mean_steps_per_p_error = 0
            mean_q_per_p_error = 0
            steps_counter = 0
            for j in range(num_of_predictions):
                num_of_steps_per_episode = 0
                prev_action = 0
                terminal_state = 0
                # generate random syndrom
                self.toric = Toric_code(self.system_size)

                if minimum_nbr_of_qubit_errors == 0:
                    self.toric.generate_random_error(p_error)
                else:
                    self.toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
                terminal_state = self.toric.terminal_state(self.toric.current_state)
                # plot one episode
                if plot_one_episode == True and j == 0 and i == 0:
                    self.toric.plot_toric_code(self.toric.current_state, 'initial_syndrom')
                
                init_qubit_state = deepcopy(self.toric.qubit_matrix)
                # solve syndrome
                while terminal_state == 1 and num_of_steps_per_episode < num_of_steps:
                    steps_counter += 1
                    num_of_steps_per_episode += 1
                    # choose greedy action
                    action, q_value = self.select_action_prediction(number_of_actions=self.number_of_actions, 
                                                                    epsilon=epsilon,
                                                                    grid_shift=self.grid_shift,
                                                                    prev_action=prev_action)
                    prev_action = action
                    self.toric.step(action)
                    self.toric.current_state = self.toric.next_state
                    terminal_state = self.toric.terminal_state(self.toric.current_state)
                    mean_q_per_p_error = incremental_mean(q_value, mean_q_per_p_error, steps_counter)
                    
                    if plot_one_episode == True and j == 0 and i == 0:
                        self.toric.plot_toric_code(self.toric.current_state, 'step_'+str(num_of_steps_per_episode))

                # compute mean steps 
                mean_steps_per_p_error = incremental_mean(num_of_steps_per_episode, mean_steps_per_p_error, j+1)
                # save error corrected 
                error_corrected[j] = self.toric.terminal_state(self.toric.current_state) # 0: error corrected # 1: error not corrected    
                # update groundstate
                self.toric.eval_ground_state()                                                          
                ground_state[j] = self.toric.ground_state # False non trivial loops

                if terminal_state == 1 or self.toric.ground_state == False:
                    failed_syndroms.append(init_qubit_state)
                    failed_syndroms.append(self.toric.qubit_matrix)

            success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
            error_corrected_list[i] = success_rate
            ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
            ground_state_list[i] =  1 - ground_state_change
            average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)
            mean_q_list[i] = np.round(mean_q_per_p_error, 3)

        return error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, failed_syndroms, ground_state_list, prediction_list_p_error, failure_rate


    def train_for_n_epochs(self, training_steps=int, epochs=int, num_of_predictions=100, num_of_steps_prediction=50, target_update=100, 
        optimizer=str, save=True, directory_path='network', prediction_list_p_error=[0.1],
        batch_size=32, replay_start_size=32, minimum_nbr_of_qubit_errors=0):
        
        data_all = []
        data_all = np.zeros((1, 19))

        for i in range(epochs):
            self.train(training_steps=training_steps,
                    target_update=target_update,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    replay_start_size=replay_start_size,
                    minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors)
            print('training done, epoch: ', i+1)
            # evaluate network
            error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, failed_syndroms, ground_state_list, prediction_list_p_error, failure_rate = self.prediction(num_of_predictions=num_of_predictions, 
                                                                                                                                                                        prediction_list_p_error=prediction_list_p_error, 
                                                                                                                                                                        minimum_nbr_of_qubit_errors=int(self.system_size/2)+1,
                                                                                                                                                                        save_prediction=True,
                                                                                                                                                                        num_of_steps=num_of_steps_prediction)

            data_all = np.append(data_all, np.array([[self.system_size, self.network_name, i+1, self.replay_memory, self.device, self.learning_rate, target_update, optimizer,
                self.discount_factor, training_steps * (i+1), mean_q_list[0], prediction_list_p_error[0], num_of_predictions, len(failed_syndroms)/2, error_corrected_list[0], ground_state_list[0], average_number_of_steps_list[0],failure_rate, self.p_error]]), axis=0)
            # save training settings in txt file 
            np.savetxt(directory_path + '/data_all.txt', data_all, 
                header='system_size, network_name, epoch, replay_memory, device, learning_rate, target_update, optimizer, discount_factor, total_training_steps, mean_q_list, prediction_list_p_error, number_of_predictions, number_of_failed_syndroms, error_corrected_list, ground_state_list, average_number_of_steps_list, failure_rate, p_error_train', delimiter=',', fmt="%s")
            # save network
            step = (i + 1) * training_steps
            PATH = directory_path + '/network_epoch/size_{3}_{2}_epoch_{0}_memory_{7}_target_update_{5}_optimizer_{6}__steps_{4}_q_{1}_discount_{8}_learning_rate_{9}.pt'.format(
                i+1, np.round(mean_q_list[0], 4), self.network_name, self.system_size, step, target_update, optimizer, self.replay_memory, self.discount_factor, self.learning_rate)
            self.save_network(PATH)
            
        return error_corrected_list
