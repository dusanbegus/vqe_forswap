# in this script we will work out a variational quantum eigensolver for transverse field Ising with h zero
# we rely heavily on QiSkit's documentation


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import efficient_su2
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QAOAAnsatz
def variational_quantum_eigensolver(dimension, h=0):
    # so ultimately, what this eigensolver will do is provide us with a ground state for the quantum Ising model
    # we use the efficient su(2) ansatz, (the basic choice). parameters are initialized randomly
    # the optimizer is taken from scipy and QiSKit's documentation
    # the first half of the code is taken directly from QiSkit's documentation
    # we will also add a magnetic field term to the Hamiltonian, and we will investigate degenerate ground states versus a single ground state

    if dimension == 2:
        hamiltonian = SparsePauliOp.from_list([("ZZ", -1.0), ("ZI", h), ("IZ", h)])
    elif dimension ==3:
        hamiltonian = SparsePauliOp.from_list([("ZZI", -1/3.0), ("IZZ", -1/3.0), ("ZIZ", -1/3.0), ("IIZ", h), ("IZI", h), ("ZII", h)])
    elif dimension ==4:
        hamiltonian = SparsePauliOp.from_list([("ZZII", -1/4.0), ("IZZI", -1/4.0), ("IIZZ", -1/4.0), ("ZIIZ", -1/4.0), ("ZIII", h), ("IZII", h), ("IIZI", h), ("IIIZ", h)])
    elif dimension ==5:
        hamiltonian = SparsePauliOp.from_list([("ZZIII", -1/5.0), ("IZZII", -1/5.0), ("IIZZI", -1/5.0), ("IIIZZ", -1/5.0), ("ZIIIZ", -1/5.0), ("ZIIII", h), ("IZIII", h), ("IIZII", h), ("IIIZI", h), ("IIIIZ", h)])
    else: 
        raise ValueError("Dimension must be between 2 and 5")
    # we note that we work with closed boundary conditions in this case
    A = np.array(hamiltonian)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    #print("The ground state energy is ", min(eigenvalues))

    # Pre-defined ansatz circuit and operator class for Hamiltonian
    
    # Note that it is more common to place initial 'h' gates outside the ansatz. Here we specifically wanted this layer structure.
    #ansatz = efficient_su2(hamiltonian.num_qubits)
    #ansatz = efficient_su2(hamiltonian.num_qubits, su2_gates=["h", "rz", "y"], entanglement="circular", reps=1)
    ansatz = efficient_su2(hamiltonian.num_qubits, su2_gates=["h", "rz", "ry","y"], reps=1)
    # we want to save the layout of the ansatz as a png
    #ansatz.draw("mpl", style="iqp").savefig(f"ansatz_layout_{dimension}_alternate_2.png")
    num_params = ansatz.num_parameters
    #print("This circuit has ", num_params, "parameters")

    #ansatz.decompose().draw("mpl", style="iqp")

    # runtime imports
    

    # To run on hardware, select the backend with the fewest number of jobs in the queue

    backend = AerSimulator()
    #print(backend)
    

    target = backend.target

    pm = generate_preset_pass_manager(target=target, optimization_level=3)

    ansatz_isa = pm.run(ansatz)

    #ansatz_isa.draw(output="mpl", idle_wires=False, style="iqp")

    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

    def cost_func(params, ansatz, hamiltonian, estimator):
        # taken from qiskit documentation
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (EstimatorV2): Estimator primitive instance
            cost_history_dict: Dictionary for storing intermediate results

        Returns:
            float: Energy estimate
        """
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]

        cost_history_dict["iters"] += 1
        cost_history_dict["prev_vector"] = params
        cost_history_dict["cost_history"].append(energy)
        #print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

        return energy

    # also taken from qiskit's documentation, with obviously some modifications
    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }
    x0 = 2 * np.pi * np.random.random(num_params)


    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 10000

        res = minimize(
            cost_func,
            x0,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method="cobyla",
            options={"maxiter": 200},
        )

    final_params = res.x
    ground_state_circuit = ansatz_isa.assign_parameters(final_params)
    # we will also want to produce the ground state
    ground_state = Statevector.from_instruction(ground_state_circuit)

    return ground_state, cost_history_dict

if __name__ == "__main__":
    np.random.seed(4+8+16+32)
    dimensions = [2,3,4,5]
    # we want to plot all the convergence histories in one plot
    plt.figure(figsize=(10, 6))

    for dimension in dimensions:
        ground_state_circuit, cost_history_dict = variational_quantum_eigensolver(dimension,h=0.0)
        plt.plot(cost_history_dict["cost_history"], label=f"Ising Model over {dimension} sites")
    plt.xlabel("Iteration (Alternative Circuit Design)")
    plt.ylabel("Energy Estimate")
    plt.title("VQE Convergence for Ising Model with varying number of sites (closed boundary conditions, alternative circuit design)")
    plt.legend()
    plt.savefig("vqe_convergence_alternative2.png")
    plt.show()