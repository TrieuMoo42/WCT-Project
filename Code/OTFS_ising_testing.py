from more_itertools import divide
from multiprocessing import Pool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time, sleep
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from scipy.linalg import circulant

from pysa.sa import Solver
from pysa.ising import get_energy

# Using 'float64' is about ~20% slower
float_type = 'float32'


# --- 1. Channel Generation Function ---
def create_sparse_dd_channel(M, N, n_paths, normalize=True, rng_seed=None):
    """
    Creates a sparse channel vector h for an M x N Delay-Doppler grid 
    with a specified number of non-zero paths.

    Args:
        M (int): Number of Delay (row) bins.
        N (int): Number of Doppler (column) bins.
        n_paths (int): The number of non-zero channel taps (paths).
        normalize (bool): If True, scales the channel so ||h||_2 = 1.
        rng_seed (int, optional): Seed for the random number generator for 
                                  reproducible results.

    Returns:
        np.array: The sparse complex channel vector h of size M*N.
    """
    
    total_bins = M * N
    if n_paths > total_bins:
        raise ValueError("Number of paths cannot exceed total number of bins (M * N).")

    # Set up random number generator
    rng = np.random.default_rng(rng_seed)

    # 1. Select the indices for the non-zero paths (DD bins)
    # This ensures the channel is sparse and the paths are located randomly.
    path_indices = rng.choice(total_bins, size=n_paths, replace=False)

    # 2. Generate complex gains for the chosen paths
    # Rayleigh fading is often modeled with IID complex Gaussian numbers.
    # We use a standard deviation of 1 for the complex components.
    path_gains_real = rng.normal(loc=0.0, scale=1.0, size=n_paths)
    path_gains_imag = rng.normal(loc=0.0, scale=1.0, size=n_paths)
    
    # Combine into complex gains
    path_gains = path_gains_real + 1j * path_gains_imag

    # 3. Assemble the full sparse channel vector
    h = np.zeros(total_bins, dtype=complex)
    h[path_indices] = path_gains

    # 4. Normalization (optional but good practice for simulation)
    if normalize:
        h = h / np.linalg.norm(h)
    
    return h


# --- 2. True OTFS Sensing Matrix Function ---
def otfs_sensing_matrix(M, N):
    """
    Generates the true phi_OTFS matrix based on 2D convolution in the DD domain,
    calculated via 2D IFFT/FFT in the TF domain.

    Assumes a single-impulse pilot at the origin (0 delay, 0 Doppler).
    """
    L = M * N
    phi_OTFS = np.zeros((L, L), dtype=complex)

    k_indices, l_indices = np.arange(L), np.arange(L)
    m_out, n_out = np.divmod(k_indices, N)
    m_in, n_in = np.divmod(l_indices, N)
    
    for k in range(L):
        for l in range(L):
            m_l, n_l = m_in[l], n_in[l] 
            m_k, n_k = m_out[k], n_out[k]
            phi_OTFS[k, l] = np.exp(-1j * 2 * np.pi * (((m_l * m_k) / M) + ((n_l * n_k) / N)))

    return phi_OTFS


# --- 3. True OTFS ML to Ising Conversion Function ---
def otfs_mle_to_ising_bipolar_model(M, N, tx_pilot, rx_pilot, n_bits=2, amp_max=1.0, sparsity_lambda=0.0):
    """
    Converts OTFS Channel Estimation (MLE) to Ising Hamiltonian parameters
    using the Bipolar Additive Offset method to handle negative gains.
    """
    
    # 1. System Model Construction (y = Phi * h)
    N_total = M * N
    y = rx_pilot
    
    # Generate the TRUE SENSING MATRIX (Phi_OTFS)
    Phi_OTFS = otfs_sensing_matrix(M, N) 
    
    # 2. Real-Valued Decomposition
    # y_real remains the same
    y_real = np.concatenate([y.real, y.imag])
    
    # Phi_real remains the same structure
    Phi_real = np.block([
        [Phi_OTFS.real, -Phi_OTFS.imag],
        [Phi_OTFS.imag,  Phi_OTFS.real]
    ])
    
    # --- BIPOLAR FIX APPLIED HERE ---
    
    # 3. Bipolar Binary Quantization Matrix (T_new)
    n_continuous_vars = 2 * N_total
    n_qubits = n_continuous_vars * n_bits

    # The weights must cover the doubled range: [0, 2 * amp_max]
    T_new = np.zeros((n_continuous_vars, n_qubits), dtype=float)
    # The weight factor is 2*amp_max * 2^-(j+1)
    weights_bipolar = np.array([2**(-i-1) for i in range(n_bits)]) * (2 * amp_max)
    
    for i in range(n_continuous_vars):
        for j in range(n_bits):
            T_new[i, i * n_bits + j] = weights_bipolar[j]

    # 4. Modify the Received Signal Vector (y_mod)
    # The offset vector 'b' has 2*MN elements, all equal to amp_max
    b_offset = amp_max * np.ones(n_continuous_vars)
    
    # We use y_mod = y_real + Phi_real @ b_offset
    y_mod = y_real + Phi_real @ b_offset

    # 5. Compute QUBO Matrices
    A_new = Phi_real @ T_new
    
    # Q_qubo = A_new.T @ A_new
    Q_qubo = np.dot(A_new.T, A_new) 
    
    # L_qubo = -2 * y_mod.T @ A_new
    L_qubo = -2 * np.dot(y_mod.T, A_new)
    
    # Add L0 sparsity penalty: L_qubo + lambda (This is an L1 penalty on x, but used as L0 approximation here)
    if sparsity_lambda > 0:
        L_qubo = L_qubo + sparsity_lambda

    # 6. Convert QUBO to Ising (x_i = (1 - s_i) / 2)
    h_ising = np.zeros(n_qubits)
    J_ising = np.zeros((n_qubits, n_qubits))
    
    # Compute J (Couplings): J_ij = Q_ij / 2.0 (for i != j)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J_val = Q_qubo[i, j] / 2.0 
            J_ising[i, j] = J_val
            J_ising[j, i] = J_val 

    # Compute h (Local Fields): h_i = -L_i/2 - sum_j Q_ij / 2.0
    for i in range(n_qubits):
        term_L = -L_qubo[i] / 2.0
        term_Q_total = -np.sum(Q_qubo[i, :]) / 2.0
        h_ising[i] = term_L + term_Q_total

    return h_ising, J_ising, amp_max # Return amp_max for demapping


# Demap best state configuration for estimated channel path gains
def demap_spin_to_channel_bipolar(s_best, M, N, n_bits, amp_max=1.0):
    """
    Decodes the optimal spin configuration (s_best) from the Ising solver 
    into the estimated complex Delay-Doppler Channel Matrix (H_DD_est) 
    using the Bipolar Additive Offset demapping.
    """
    
    L = M * N
    N_continuous_vars = 2 * L
    
    # 1. Spin to Binary Variable (s_best -> x_best)
    x_best = (1 - s_best) / 2
    
    # 2. Binary to Continuous Variable (x_best -> h_shifted_est)
    
    # The new weights_bipolar must be used, corresponding to a range of 2*amp_max
    weights_bipolar = np.array([2**(-i-1) for i in range(n_bits)]) * (2 * amp_max)
    
    h_shifted_est = np.zeros(N_continuous_vars, dtype=float)
    
    for i in range(N_continuous_vars):
        start_idx = i * n_bits
        end_idx = start_idx + n_bits
        binary_segment = x_best[start_idx:end_idx]
        
        # h_shifted_est is the UNIPOLAR estimation
        h_shifted_est[i] = np.dot(binary_segment, weights_bipolar)
        
    # --- BIPOLAR FIX APPLIED HERE ---
    
    # 3. Apply the Bipolar Offset: h_real = h_shifted - amp_max
    h_real_est = h_shifted_est - amp_max
    
    # 4. Continuous to Complex Matrix (h_real_est -> H_DD_est)
    
    # Separate the real and imaginary parts
    h_real = h_real_est[:L]
    h_imag = h_real_est[L:]
    
    # Combine to form the complex channel tap vector
    h_est_vector = h_real + 1j * h_imag
    
    # Reshape the vector into the final M x N Delay-Doppler channel matrix
    H_DD_est = h_est_vector.reshape((M, N))
    
    return H_DD_est


# Compare estimated channel with true channel (NMSE)
def compare_channel_matrices(H_true, H_est):
    """
    Compares the estimated complex channel matrix (H_est) against the 
    true complex channel matrix (H_true) using two metrics:
    1. Mean Squared Error (MSE)
    2. Normalized Mean Squared Error (NMSE) in dB

    Args:
        H_true (np.array): The true M x N complex channel matrix.
        H_est (np.array): The estimated M x N complex channel matrix.

    Returns:
        dict: A dictionary containing 'MSE' and 'NMSE_dB'.
    """
    # 1. Vectorize the matrices to calculate the Frobenius (L2) norm
    h_true_vector = H_true.flatten()
    h_est_vector = H_est.flatten()

    # 2. Calculate the Error Vector
    error_vector = h_true_vector - h_est_vector

    # 3. Calculate the Sum of Squared Error (Numerator of NMSE)
    # ||h_true - h_est||_2^2
    squared_error_norm = np.sum(np.abs(error_vector)**2)

    # 4. Calculate the Energy of the True Channel (Denominator of NMSE)
    # ||h_true||_2^2
    true_channel_norm = np.sum(np.abs(h_true_vector)**2)

    # 5. Calculate MSE
    # MSE = Sum of Squared Error / Total Number of Taps
    mse = squared_error_norm / h_true_vector.size

    # 6. Calculate NMSE (in dB)
    if true_channel_norm == 0:
        # Handle the edge case where the true channel is all zeros (shouldn't happen in OTFS)
        nmse_db = np.nan
    else:
        nmse = squared_error_norm / true_channel_norm
        nmse_db = 10 * np.log10(nmse)
    
    return {
        "MSE": mse,
        "NMSE_dB": nmse_db
    }


def calculate_snr(Y_ideal, sigma_noise, M, N):
    """
    Calculates the Signal-to-Noise Ratio (SNR) for an OTFS system.

    SNR is defined as the ratio of the average energy of the ideal received signal
    (Y_ideal) to the average power of the added noise.

    Args:
        Y_ideal (np.array): The M*N x 1 complex vector of the received signal
                            BEFORE noise is added.
        sigma_noise (float): The standard deviation (RMS value) of the complex AWGN.
                             (Assumes noise is AWGN: N = n_real + j*n_imag, 
                             where E[|N|^2] = 2 * sigma_noise^2, but we use the
                             commonly accepted convention for total complex noise power).
        M (int): Delay dimension.
        N (int): Doppler dimension.

    Returns:
        dict: A dictionary containing 'SNR_linear' and 'SNR_dB'.
    """
    L = M * N
    
    # --- 1. Signal Power Calculation (Average Power) ---
    # Signal Power = Average of the squared magnitude of the ideal received signal.
    # ||Y_ideal||_2^2 / L
    signal_power = np.sum(np.abs(Y_ideal)**2) / L
    
    # --- 2. Noise Power Calculation (Average Power) ---
    # The average power of complex AWGN where the real and imaginary parts 
    # each have variance sigma_noise^2 is P_Noise = 2 * sigma_noise^2.
    # However, in many telecom simulations, the input sigma_noise is normalized
    # such that the total complex noise power is sigma_noise^2. We use the 
    # standard calculation based on the variance of the real/imaginary components.
    
    # Assume the code adding noise uses: 
    # noise = sigma_noise * (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)
    # If the noise generation is standard (Re and Im parts have variance 0.5 * P_noise),
    # the total average noise power is simply P_Noise = 2 * (sigma_noise^2 / 2) = sigma_noise^2.
    
    # We use the standard theoretical result for complex AWGN power:
    noise_power = 2 * sigma_noise**2
    
    # --- 3. SNR Calculation ---
    if noise_power == 0:
        # Avoid division by zero (occurs when sigma_noise = 0)
        snr_linear = np.inf
        snr_db = np.inf
    else:
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

    return {
        "SNR_linear": snr_linear,
        "SNR_dB": snr_db
    }


if __name__ == "__main__":
    # Number of delay(M)-Doppler(N) bins
    M_dim = 4
    N_dim = 4
    L_total = M_dim * N_dim

    # Number of bits for gain quantization
    n_bits = 4

    # Number of paths
    num_paths = 3
    
    # Generate Channel (h_true)
    h_true = create_sparse_dd_channel(M_dim, N_dim, num_paths, rng_seed=42)

    # Generate Pilot (tx) and Received Signal (rx)
    tx = np.zeros((M_dim,N_dim), dtype=complex)
    tx[0,0] = 1.0 + 0j

    Phi_OTFS = otfs_sensing_matrix(M_dim, N_dim)

    # Initialize lists to store results across all SNR points
    all_mean_nmse = []
    all_lower_nmse = [] # 5th percentile
    all_upper_nmse = [] # 95th percentile

    sigma_noise_range = np.arange(0.01, 0.51, 0.01) # 0.00 to 0.50

    Y_ideal = np.dot(Phi_OTFS, h_true)
    snr_range = []

    for sigma in sigma_noise_range:
        # Calculate the SNR for the current noise level
        snr_metrics = calculate_snr(Y_ideal, sigma, M_dim, N_dim)
        
        # Store the SNR for the X-axis
        snr_range.append(snr_metrics['SNR_dB'])

    for j in sigma_noise_range:
        rx = Phi_OTFS @ h_true + j*(np.random.randn(L_total) + 1j * np.random.randn(L_total))

        # Using n bits per real number, so we expect L_bins * 2(re/im) * n_bits = #(qubits)
        # Convert to Ising
        h, J, amp_max_val = otfs_mle_to_ising_bipolar_model(
            M_dim, N_dim, 
            tx, rx, 
            n_bits=n_bits, 
            sparsity_lambda=0.5
        )

        # Your data
        # h: shape (n_vars,)
        # J: shape (n_vars, n_vars), symmetric
        n_vars = h.shape[0]

        # Start from your coupling matrix
        problem = J.astype(np.float32).copy()

        # Add local fields to the diagonal
        for i in range(n_vars):
            problem[i, i] += h[i]

        float_type = "float32"
        solver = Solver(problem=problem, problem_type="ising", float_type=float_type)

        # Number of variables
        n_sweeps = 100
        n_replicas = 2
        n_reads = 1000
        min_temp = 0.1
        max_temp = 0.3

        # Apply Metropolis, same initialization
        res_1 = solver.metropolis_update(
            num_sweeps=n_sweeps,
            num_reads=n_reads,
            num_replicas=n_replicas,
            update_strategy='sequential',
            min_temp=min_temp,
            max_temp=max_temp,
            initialize_strategy='ones',
            recompute_energy=True,
            sort_output_temps=True,
            parallel=True,  # True by default
            verbose=False)
        
        # Extract ALL states returned by the solver (all 'num_reads' states)
        s_best_states_all = res_1['best_state']

        nmse_readings = []
        for k in range(len(s_best_states_all)):
            # Get the current spin configuration for this read
            s_current = s_best_states_all.iloc[k]
            
            # Demap the spin configuration
            H_est = demap_spin_to_channel_bipolar(s_current, M_dim, N_dim, n_bits)
            
            # Calculate NMSE
            metrics = compare_channel_matrices(h_true.reshape((M_dim,N_dim)), H_est)
            nmse_readings.append(metrics['NMSE_dB'])

        # Convert to numpy array for statistics
        nmse_readings = np.array(nmse_readings)

        # Calculate statistics for the confidence interval
        mean_nmse_current = np.mean(nmse_readings)
        lower_nmse_current = np.percentile(nmse_readings, 5)  # 5th percentile
        upper_nmse_current = np.percentile(nmse_readings, 95) # 95th percentile

        # Store the results for plotting later
        all_mean_nmse.append(mean_nmse_current)
        all_lower_nmse.append(lower_nmse_current)
        all_upper_nmse.append(upper_nmse_current)

    plt.figure(figsize=(10, 6))

    # Plot the mean NMSE line
    plt.plot(snr_range, all_mean_nmse, marker='o', linestyle='-', color='blue', label='Mean NMSE')

    # # Plot the 90% percentile range (5th to 95th percentile)
    # plt.fill_between(snr_range, all_lower_nmse, all_upper_nmse, color='blue', alpha=0.2, label='90% Percentile Range')

    plt.title(f'NMSE vs. SNR')
    plt.xlabel('Signal-to-Noise Ratio (SNR) in dB')
    plt.ylabel('Normalized Mean Squared Error (NMSE) in dB')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()