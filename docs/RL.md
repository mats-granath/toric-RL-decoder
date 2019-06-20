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

## load_network
  Loads a already trained network from a given path defined in a string, PATH

##
