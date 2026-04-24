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

def initialize_circuit(state, basis, dimension, shotss=1000):
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
def varying_shots():
    dimension=4
    shots=[100, 1000, 2000, 5000, 10000]
    repeats=5
    errors=[]
    stds=[]
    for shot in shots:
        err=[]
        for r in range(repeats):
            b=basiss(dimension)
            target=create_rand(2**dimension)
            c_values, s2_values=initialize_circuit(target, b, dimension, shotss=shot)
            err.append(np.abs(np.sum(s2_values)-1.0))
        errors.append(np.mean(err))
        stds.append(np.std(err))
    # we will also add the logarithmic plot inside the main plot
    log_errors=[np.log(e) for e in errors]
    log_shots=[np.log(s) for s in shots]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(shots, errors, label="Error", color='tab:blue')
    ax1.fill_between(shots, np.array(errors) - np.array(stds), 
                    np.array(errors) + np.array(stds), alpha=0.2, 
                    label="Error standard deviation", color='tab:blue')

    ax1.set_xlabel("Number of shots")
    ax1.set_ylabel("Error")
    ax1.set_title("SWAP test error vs Number of shots: 4 qubits")
    ax1.legend(loc='lower left') 
    # copilot suggested this
    inset_ax = fig.add_axes([0.55, 0.55, 0.3, 0.3]) 

    # Plot Log Data on the inset
    # Note: You asked for Log B vs A (Log Error vs Shots)
    inset_ax.plot(log_shots, log_errors, color='tab:red')
    inset_ax.set_title("Log Error Plot", fontsize=10)
    inset_ax.set_xlabel("Log(Shots)", fontsize=8)
    inset_ax.set_ylabel("Log(Error)", fontsize=8)
    inset_ax.tick_params(labelsize=8) # Make tick numbers smaller

    # 3. Save and Show
    plt.savefig("swap_test_error_shots.png", dpi=300)
    plt.show()
def the_test():
    dimensions=[2,3,4,5]
    repeats=30 # repeat the experiment 30 times for each dimension and average the error
    err=[]
    var=[]
    for dimension in dimensions:
        errors=[]
        for r in range(repeats):
            b=basiss(dimension)
            target=create_rand(2**dimension)
            c_values, s2_values=initialize_circuit(target, b, dimension)
            errors.append(np.abs(np.sum(s2_values)-1.0) )
        err.append(np.mean(errors))
        var.append(np.var(errors))      
    # now we plot the errors against the dimension of the Hilbert space, along with variances                            
    Ns=[2**d for d in dimensions]
    plt.plot(Ns, err, label="Average Error")
    plt.fill_between(Ns, np.array(err) - np.array(var), np.array(err) + np.array(var), alpha=0.2, label="Error Standard Deviation")
    plt.xlabel("Hilbert space size (N)")
    plt.ylabel("Average Error")
    plt.title("SWAP Test Error vs Number of Ground States")
    plt.legend()
    plt.savefig("swap_test_error.png")
    plt.show()


if __name__ == "__main__":
    the_test()
