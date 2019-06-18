import numpy
from collections import namedtuple
import random
import _pickle as cPickle
from sum_tree import Sum_tree


Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'terminal'])


class Replay_memory_uniform(object):

    def __init__(self, capacity): # alpha is not in use 
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def save(self, data, priority): # priority is not used 
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta): # beta is not used 
        if len(self.memory) < batch_size:
            return
        return random.sample(self.memory, batch_size), None, None # added for prioritized replay memory

    def save_memory(self, size):
        file = open('replay_memory/replay_memory_size_'+str(size)+'_capacity_'+str(self.capacity)+'.txt','wb+')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load_memory(self, name):
        file = open('replay_memory/'+str(name)+'.txt' ,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)

    def __len__(self):
        return len(self.memory)


class Replay_memory_prioritized(object):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self, memory_size, alpha):
        """ Prioritized experience replay buffer initialization.
        
        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = Sum_tree(memory_size)
        self.memory_size = memory_size
        #self.batch_size = batch_size
        self.alpha = alpha

    def save(self, data, priority):
        """ Add new sample.
        
        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority**self.alpha)

    def sample(self, batch_size, beta):
        """ The method return samples randomly.
        
        Parameters
        ----------
        beta : float
        
        Returns
        -------
        out : 
            list of samples
        weights: 
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """
        
        if self.tree.filled_size() < batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0]) # To avoid duplicating
            
        
        self.priority_update(indices, priorities) # Revert priorities
        #print(weights)
        weights = [ i/max(weights) for i in weights] # Normalize for stability
        
        return out, weights, indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.
        
        Parameters
        ----------
        indices : 
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)
    
    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)


    def print_tree(self):
        self.tree.print_tree()    
            
        
        
