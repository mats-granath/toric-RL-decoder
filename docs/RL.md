# RL
  This class implements a reinforcement learning agent that decodes errors on the toric code. The agent make use of deep reinforcement learning to learn how to correct errors that can appear on the toric code.

## __init__ 
    notable parameters:
      Network: a pytorch neural network
      Network: a string with the Network name
      system_size: the grid size of the toric code the agent will be used on, Only accepts ODD gridsizes.
      p_error: the error probability of an qubit when generating random syndromes.
      number_of_actions: the number of possible operators that can be used on a qubit, e.g. 3 corresponds to the sigma^x, sigma^z and sigma^y operators.
      device: a string that describes what device to use during forward and backpropagation. Valid values: 'cpu' and 'gpu'.
      replay_memory: string describing the type of replay memory used. Valid values: 'uniform' and 'proportional'.

## save_network(self, PATH)
  A function that saves a trained network at a given path defined in a string, PATH.

## load_network(self, PATH)
  Loads a already trained network from a given path defined in a string, PATH

## experience_replay

## train
  A function used for training of the policy network.
  notable parameters:
    training_steps: number of training iterations the algorithm takes
    target_update: the frequency the target network is updated

## prediction
  Function used for evaluating the training of the network, the algorithm is tested on randomly generated syndromes.
  notable parameters:
    num_of_steps: maximum number of steps allowed on one syndrome.
    PATH: string describing file-path of an trained model, if left blanck de current model of the class is used.

## train_for_n_epochs
  Function that trains the model a number of epochs. Each epoch consist of training during a number of steps and prediction, i.e. evaluation, for a number of syndromes.
  notable parameters:
    training_steps: nbr of training iterations the algorithm uses each epoch    epochs: number of epochs the training will use.
