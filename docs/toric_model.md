# toric_model

![](src/toric_code_gif.gif)

This class includes all the functions to initialize, alter and display the toric code.  

## Constructor 
- system_size: the size of the 2d grid (see gif d=5)
- plaquette_matrix: dxd matrix containing the excitations in the plaquette space (int)
- vertex_matrix: dxd matrix containing the excitations in the vertex space (int)
- qubit_matrix: The 2xdxd matrix is initialized with only 0 (ground state without trivial and non trivial loops). It contains the current qubit state. Identity = 0, pauli_x = 1, pauli_y = 2, pauli_z = 3.
- state: 2xdxd matrix of vertex and plaquette matrix of current state
- next_state: 2xdxd matrix of vertex and plaquette matrix of the next state (after taking an action)
- ground_state: boolean: True, ground state is conserved error correction successful; False, non trivial loops, error correction failed.
- rule_table: covering the interaction of different pauli operators acting on the same qubit. 

## Functions
