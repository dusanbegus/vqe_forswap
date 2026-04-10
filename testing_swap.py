import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sklearn
import sys
from qiskit import QuantumCircuit
from qiskit.circuit.library import CSwapGate
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator 
import setup

# dimension indicates the number of qubits we are working with


def basiss(dimension):
    basis_states=[]
    for i in range(2**dimension):
        state=[0 for _ in range(2**dimension)]
        state[i]=1
        basis_states.append(state)

    return basis_states

def create_rand(dimension):
    state=np.random.rand(dimension)
    phases=np.random.rand(dimension)*2*np.pi
    state=state*np.exp(1j*phases)
    state=state/np.linalg.norm(state)
    return np.array(state)

def initialize_circuit(state, basis, dimension):
    cs=[]
    S2=[]
    for vector in basis:
        qc=QuantumCircuit(2*dimension +1)
        # we will have to compare the target and the basis via a SWAP test
        first=[]
        for i in range(dimension):
            first.append(i)
        second=[]
        for i in range(dimension, 2*dimension):
            second.append(i)
        qc.initialize(state, first, normalize=True)
        qc.initialize(vector, second, normalize=True)  
        
        # now we will compute the fidelity between these two states
        qc.initialize([1,0],2*dimension, normalize=True)
        # this is the extra qubit storing the information about the overlap

        qc.add_register(ClassicalRegister(1, 'c'))
        qc.h(2*dimension)

        for i in range(dimension):
            qc.append(CSwapGate(), [2*dimension, first[i], second[i]])
        qc.h(2*dimension) 

        simulator = AerSimulator(method='statevector')   
        qc.measure(2*dimension,0)
        # so now we are done with the SWAP test
        shotss=1000
        job = simulator.run(qc, shots=shotss)  # Run 100 times
        result = job.result()
        counts = result.get_counts(qc)
        num_zeros = counts.get('0', 0)
        S2.append(np.abs(2*(num_zeros / shotss)-1.0))
        cs.append(np.sqrt(np.abs(2*(num_zeros / shotss)-1.0)))
    # normalize cs
    S2=np.array(S2)
    cs=np.array(cs)
    return cs,S2
def the_test():
    dimensions=[2,3,4,5]
    repeats=30 # repeat the experiment 30 times for each dimension and average the error
    err=[]
    for dimension in dimensions:
        errors=[]
        for r in range(repeats):
            b=basiss(dimension)
            target=create_rand(2**dimension)
            c_values, s2_values=initialize_circuit(target, b, dimension)
            errors.append(np.sum(s2_values)-1.0) 
        err.append(np.mean(errors))
    Ns=[2**d for d in dimensions]
    plt.plot(Ns, err)
    plt.xlabel("Hilbert space size (N)")
    plt.ylabel("Average Error")
    plt.title("SWAP Test Error vs Hilbert Space ize")
    plt.savefig("swap_test_error.png")
    plt.show()


if __name__ == "__main__":
    the_test()
