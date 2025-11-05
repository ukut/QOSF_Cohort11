import numpy as np
from typing import List, Union, Tuple
import math


# ============================================================================
# CORE STATE PREPARATION FUNCTIONS
# ============================================================================

def normalize_amplitudes(amplitudes: Union[List[complex], np.ndarray]) -> np.ndarray:
    """
    Normalize a list of complex amplitudes to unit norm.
    
    For a quantum state |ψ⟩ = Σ aᵢ|i⟩, normalization requires:
    ⟨ψ|ψ⟩ = Σ |aᵢ|² = 1
    
    If the input is already normalized, return it as is.
    If the input is a zero vector, raise an error.
    
    Args:
        amplitudes: List or array of complex numbers
        
    Returns:
        Normalized NumPy array of complex numbers
        
    Raises:
        ValueError: If all amplitudes are zero
    """
    amps = np.array(amplitudes, dtype=complex)
    
    # Compute the L2 norm: ||a|| = sqrt(Σ |aᵢ|²)
    norm_squared = np.sum(np.abs(amps) ** 2)
    norm = np.sqrt(norm_squared)
    
    if np.isclose(norm, 0.0):
        raise ValueError("Cannot normalize: all amplitudes are zero")
    
    # Normalize: aᵢ_normalized = aᵢ / ||a||
    normalized = amps / norm
    
    return normalized


def verify_normalization(amplitudes: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Verify that a state vector is normalized: Σ |aᵢ|² = 1
    
    Args:
        amplitudes: Complex array representing quantum state
        tolerance: Allowed deviation from unit norm
        
    Returns:
        True if normalized within tolerance, False otherwise
    """
    norm_squared = np.sum(np.abs(amplitudes) ** 2)
    return np.isclose(norm_squared, 1.0, atol=tolerance)


def prepare_two_qubit_state(amplitudes: Union[List[complex], np.ndarray]) -> np.ndarray:
    """
    Prepare a two-qubit quantum state vector from four complex amplitudes.
    
    The two-qubit state is represented as:
        |ψ⟩ = a₀|00⟩ + a₁|01⟩ + a₂|10⟩ + a₃|11⟩
    
    Where:
    - |00⟩ = [1, 0, 0, 0]ᵀ (both qubits in state |0⟩)
    - |01⟩ = [0, 1, 0, 0]ᵀ (first qubit |0⟩, second qubit |1⟩)
    - |10⟩ = [0, 0, 1, 0]ᵀ (first qubit |1⟩, second qubit |0⟩)
    - |11⟩ = [0, 0, 0, 1]ᵀ (both qubits in state |1⟩)
    
    The state vector is the direct stacking of normalized amplitudes.
    
    Args:
        amplitudes: List or array of exactly 4 complex numbers [a₀, a₁, a₂, a₃]
        
    Returns:
        NumPy array of shape (4,) containing the normalized state vector
        
    Raises:
        ValueError: If not exactly 4 amplitudes are provided
        ValueError: If all amplitudes are zero
    """
    # Validate input size
    amps = np.array(amplitudes, dtype=complex)
    if amps.size != 4:
        raise ValueError(f"Expected 4 amplitudes for 2-qubit state, got {amps.size}")
    
    # Normalize amplitudes
    normalized_amps = normalize_amplitudes(amps)
    
    # Return as 1D state vector
    return normalized_amps


def prepare_three_qubit_state(amplitudes: Union[List[complex], np.ndarray]) -> np.ndarray:
    """
    Prepare a three-qubit quantum state vector from eight complex amplitudes.
    
    The three-qubit state is represented as:
        |ψ⟩ = a₀|000⟩ + a₁|001⟩ + a₂|010⟩ + a₃|011⟩ + 
               a₄|100⟩ + a₅|101⟩ + a₆|110⟩ + a₇|111⟩
    
    The state vector has dimension 2³ = 8.
    
    Args:
        amplitudes: List or array of exactly 8 complex numbers [a₀, ..., a₇]
        
    Returns:
        NumPy array of shape (8,) containing the normalized state vector
        
    Raises:
        ValueError: If not exactly 8 amplitudes are provided
        ValueError: If all amplitudes are zero
    """
    # Validate input size
    amps = np.array(amplitudes, dtype=complex)
    if amps.size != 8:
        raise ValueError(f"Expected 8 amplitudes for 3-qubit state, got {amps.size}")
    
    # Normalize amplitudes
    normalized_amps = normalize_amplitudes(amps)
    
    # Return as 1D state vector
    return normalized_amps


def prepare_n_qubit_state(amplitudes: Union[List[complex], np.ndarray], 
                          n_qubits: int) -> np.ndarray:
    """
    Generalized function to prepare an n-qubit quantum state vector.
    
    The n-qubit state has dimension 2ⁿ.
    
    Args:
        amplitudes: List or array of 2ⁿ complex numbers
        n_qubits: Number of qubits (state dimension = 2^n_qubits)
        
    Returns:
        NumPy array of shape (2^n_qubits,) containing the normalized state vector
        
    Raises:
        ValueError: If the number of amplitudes doesn't match 2ⁿ
        ValueError: If all amplitudes are zero
    """
    expected_size = 2 ** n_qubits
    
    amps = np.array(amplitudes, dtype=complex)
    if amps.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} amplitudes for {n_qubits}-qubit state, "
            f"got {amps.size}"
        )
    
    # Normalize amplitudes
    normalized_amps = normalize_amplitudes(amps)
    
    return normalized_amps


# ============================================================================
# TENSOR PRODUCT OPERATIONS (For educational purposes)
# ============================================================================

def tensor_product(state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
    """
    Compute the tensor product (Kronecker product) of two state vectors.
    
    For single-qubit states |ψ⟩ and |φ⟩:
        |ψ⟩ ⊗ |φ⟩
    
    Args:
        state1: First state vector
        state2: Second state vector
        
    Returns:
        Combined state vector with dimension (state1.size * state2.size)
    """
    return np.kron(state1, state2)


def construct_two_qubit_from_tensor_product(amplitudes: Union[List[complex], np.ndarray]) -> np.ndarray:
    """
    Construct a two-qubit state by explicitly interpreting amplitudes as
    coefficients in the computational basis and stacking them.
    
    This demonstrates the direct stacking approach:
    |ψ⟩ = [a₀, a₁, a₂, a₃]ᵀ where aᵢ are normalized amplitudes
    
    Args:
        amplitudes: 4 complex amplitudes
        
    Returns:
        2-qubit state vector
    """
    amps = np.array(amplitudes, dtype=complex)
    if amps.size != 4:
        raise ValueError(f"Expected 4 amplitudes, got {amps.size}")
    
    normalized = normalize_amplitudes(amps)
    
    # The state vector is simply the stacked normalized amplitudes
    # in the computational basis {|00⟩, |01⟩, |10⟩, |11⟩}
    return normalized.reshape(-1)  # Ensure 1D


# ============================================================================
# MEASUREMENT & ANALYSIS FUNCTIONS
# ============================================================================

def get_measurement_probabilities(state_vector: np.ndarray) -> np.ndarray:
    """
    Compute measurement probabilities for each computational basis state.
    
    For a state |ψ⟩ = Σ aᵢ|i⟩, the probability of measuring outcome |i⟩ is:
        P(i) = |aᵢ|²
    
    Args:
        state_vector: Normalized quantum state vector
        
    Returns:
        Array of probabilities (non-negative, sum to 1)
    """
    probabilities = np.abs(state_vector) ** 2
    return probabilities


def get_basis_labels(n_qubits: int) -> List[str]:
    """
    Generate computational basis labels for n qubits.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        List of basis state labels, e.g., ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    """
    basis_size = 2 ** n_qubits
    labels = []
    for i in range(basis_size):
        binary = format(i, f'0{n_qubits}b')
        label = f"|{binary}⟩"
        labels.append(label)
    return labels


def print_state_info(state_vector: np.ndarray, n_qubits: int, title: str = "Quantum State"):
    """
    Print detailed information about a quantum state.
    
    Args:
        state_vector: Normalized quantum state vector
        n_qubits: Number of qubits
        title: Title for the output
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    labels = get_basis_labels(n_qubits)
    probabilities = get_measurement_probabilities(state_vector)
    
    print(f"\nState vector: {state_vector}")
    print(f"\nAmplitudes and probabilities:")
    
    for label, amplitude, probability in zip(labels, state_vector, probabilities):
        if probability > 1e-10:  # Only print non-negligible probabilities
            print(f"  {label}: amplitude = {amplitude:+.6f}, P = {probability:.6f}")
    
    print(f"\nNormalization check: Σ|aᵢ|² = {np.sum(probabilities):.10f}")
    print(f"Is normalized: {verify_normalization(state_vector)}")


# ============================================================================
# UNIT TESTS
# ============================================================================

import unittest


class TestQuantumStatePreparation(unittest.TestCase):
    """Comprehensive unit tests for quantum state preparation."""
    
    # ========================================================================
    # Tests for normalization
    # ========================================================================
    
    def test_normalize_amplitudes_already_normalized(self):
        """Test normalization of already-normalized amplitudes."""
        amplitudes = [1/np.sqrt(2), 1/np.sqrt(2), 0, 0]
        normalized = normalize_amplitudes(amplitudes)
        
        # Check that result is normalized
        norm_squared = np.sum(np.abs(normalized) ** 2)
        self.assertAlmostEqual(norm_squared, 1.0, places=10)
    
    def test_normalize_amplitudes_unnormalized(self):
        """Test normalization of unnormalized amplitudes."""
        amplitudes = [1, 1, 1, 1]
        normalized = normalize_amplitudes(amplitudes)
        
        # Manually check normalization
        norm_squared = np.sum(np.abs(normalized) ** 2)
        self.assertAlmostEqual(norm_squared, 1.0, places=10)
        
        # All amplitudes should be 0.5 (since 1/√4 = 0.5)
        expected = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_normalize_amplitudes_complex(self):
        """Test normalization of complex amplitudes."""
        amplitudes = [1+1j, 1-1j, 0, 0]
        normalized = normalize_amplitudes(amplitudes)
        
        # Check normalization
        norm_squared = np.sum(np.abs(normalized) ** 2)
        self.assertAlmostEqual(norm_squared, 1.0, places=10)
    
    def test_normalize_amplitudes_zero_vector(self):
        """Test that normalizing a zero vector raises an error."""
        amplitudes = [0, 0, 0, 0]
        with self.assertRaises(ValueError):
            normalize_amplitudes(amplitudes)
    
    def test_verify_normalization_normalized(self):
        """Test verification of a normalized state."""
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
        self.assertTrue(verify_normalization(state))
    
    def test_verify_normalization_unnormalized(self):
        """Test verification of an unnormalized state."""
        state = np.array([1, 1, 0, 0], dtype=complex)
        self.assertFalse(verify_normalization(state))
    
    # ========================================================================
    # Tests for two-qubit state preparation
    # ========================================================================
    
    def test_prepare_two_qubit_state_correct_dimension(self):
        """Test that 2-qubit state has dimension 4."""
        state = prepare_two_qubit_state([1, 0, 0, 0])
        self.assertEqual(state.shape, (4,))
        self.assertEqual(state.size, 4)
    
    def test_prepare_two_qubit_state_normalized(self):
        """Test that 2-qubit state is normalized."""
        amplitudes = [1, 1, 1, 1]
        state = prepare_two_qubit_state(amplitudes)
        self.assertTrue(verify_normalization(state))
    
    def test_prepare_two_qubit_state_basis_00(self):
        """Test preparation of |00⟩ state."""
        state = prepare_two_qubit_state([1, 0, 0, 0])
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(state, expected)
    
    def test_prepare_two_qubit_state_basis_11(self):
        """Test preparation of |11⟩ state."""
        state = prepare_two_qubit_state([0, 0, 0, 1])
        expected = np.array([0, 0, 0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(state, expected)
    
    def test_prepare_two_qubit_state_equal_superposition(self):
        """Test preparation of equal superposition (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2."""
        state = prepare_two_qubit_state([1, 1, 1, 1])
        expected = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        np.testing.assert_array_almost_equal(state, expected)
    
    def test_prepare_two_qubit_state_bell_state(self):
        """Test preparation of Bell state (|00⟩ + |11⟩)/√2."""
        state = prepare_two_qubit_state([1, 0, 0, 1])
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(state, expected)
    
    def test_prepare_two_qubit_state_complex_amplitudes(self):
        """Test preparation with complex amplitudes."""
        # (1 + 1j|01⟩ + (-1)|10⟩ + (1+1j)|11⟩) / norm
        amplitudes = [0, 1+1j, -1, 1+1j]
        state = prepare_two_qubit_state(amplitudes)
        
        # Verify normalization
        self.assertTrue(verify_normalization(state))
        
        # Check dimensions
        self.assertEqual(state.shape, (4,))
    
    def test_prepare_two_qubit_state_invalid_input_size(self):
        """Test that invalid input size raises an error."""
        with self.assertRaises(ValueError):
            prepare_two_qubit_state([1, 0, 0])  # Only 3 amplitudes
        
        with self.assertRaises(ValueError):
            prepare_two_qubit_state([1, 0, 0, 0, 1])  # 5 amplitudes
    
    def test_prepare_two_qubit_state_unnormalized_input(self):
        """Test that unnormalized input is properly normalized."""
        unnormalized = [2, 0, 0, 0]
        state = prepare_two_qubit_state(unnormalized)
        
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(state, expected)
    
    # ========================================================================
    # Tests for three-qubit state preparation
    # ========================================================================
    
    def test_prepare_three_qubit_state_correct_dimension(self):
        """Test that 3-qubit state has dimension 8."""
        state = prepare_three_qubit_state([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(state.shape, (8,))
        self.assertEqual(state.size, 8)
    
    def test_prepare_three_qubit_state_normalized(self):
        """Test that 3-qubit state is normalized."""
        amplitudes = [1] * 8
        state = prepare_three_qubit_state(amplitudes)
        self.assertTrue(verify_normalization(state))
    
    def test_prepare_three_qubit_state_basis_000(self):
        """Test preparation of |000⟩ state."""
        state = prepare_three_qubit_state([1, 0, 0, 0, 0, 0, 0, 0])
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(state, expected)
    
    def test_prepare_three_qubit_state_basis_111(self):
        """Test preparation of |111⟩ state."""
        state = prepare_three_qubit_state([0, 0, 0, 0, 0, 0, 0, 1])
        expected = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(state, expected)
    
    def test_prepare_three_qubit_state_equal_superposition(self):
        """Test preparation of equal superposition of all basis states."""
        state = prepare_three_qubit_state([1] * 8)
        expected = np.array([1/np.sqrt(8)] * 8, dtype=complex)
        np.testing.assert_array_almost_equal(state, expected)
    
    def test_prepare_three_qubit_state_invalid_input_size(self):
        """Test that invalid input size raises an error."""
        with self.assertRaises(ValueError):
            prepare_three_qubit_state([1, 0, 0, 0, 0, 0, 0])  # 7 amplitudes
        
        with self.assertRaises(ValueError):
            prepare_three_qubit_state([1] * 9)  # 9 amplitudes
    
    # ========================================================================
    # Tests for n-qubit state preparation
    # ========================================================================
    
    def test_prepare_n_qubit_state_1_qubit(self):
        """Test n-qubit state preparation for 1 qubit."""
        state = prepare_n_qubit_state([1, 0], n_qubits=1)
        self.assertEqual(state.size, 2)
        self.assertTrue(verify_normalization(state))
    
    def test_prepare_n_qubit_state_4_qubits(self):
        """Test n-qubit state preparation for 4 qubits."""
        amplitudes = [1] * 16  # 2^4 = 16
        state = prepare_n_qubit_state(amplitudes, n_qubits=4)
        self.assertEqual(state.size, 16)
        self.assertTrue(verify_normalization(state))
    
    def test_prepare_n_qubit_state_invalid_size(self):
        """Test that invalid size raises an error."""
        with self.assertRaises(ValueError):
            prepare_n_qubit_state([1, 0, 0], n_qubits=2)  # Expected 4
    
    # ========================================================================
    # Tests for measurement probabilities
    # ========================================================================
    
    def test_get_measurement_probabilities_sum_to_one(self):
        """Test that probabilities sum to 1."""
        state = prepare_two_qubit_state([1, 1, 1, 1])
        probs = get_measurement_probabilities(state)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=10)
    
    def test_get_measurement_probabilities_values(self):
        """Test that probability values are non-negative and ≤ 1."""
        state = prepare_two_qubit_state([1+1j, 0, 1j, 1])
        probs = get_measurement_probabilities(state)
        
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))
    
    def test_get_measurement_probabilities_basis_state(self):
        """Test probabilities for a basis state."""
        state = prepare_two_qubit_state([1, 0, 0, 0])
        probs = get_measurement_probabilities(state)
        
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_array_almost_equal(probs, expected)
    
    # ========================================================================
    # Tests for tensor product operations
    # ========================================================================
    
    def test_tensor_product_dimensions(self):
        """Test that tensor product has correct dimension."""
        state1 = np.array([1, 0], dtype=complex)
        state2 = np.array([1, 0], dtype=complex)
        result = tensor_product(state1, state2)
        
        self.assertEqual(result.size, 4)
    
    def test_tensor_product_basis_00(self):
        """Test tensor product |0⟩ ⊗ |0⟩ = |00⟩."""
        ket_0 = np.array([1, 0], dtype=complex)
        result = tensor_product(ket_0, ket_0)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_tensor_product_superposition(self):
        """Test tensor product of superposition states."""
        # (|0⟩ + |1⟩)/√2 ⊗ |0⟩
        superposition = np.array([1, 1], dtype=complex) / np.sqrt(2)
        ket_0 = np.array([1, 0], dtype=complex)
        
        result = tensor_product(superposition, ket_0)
        expected = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    # ========================================================================
    # Integration tests
    # ========================================================================
    
    def test_integration_various_amplitudes(self):
        """Integration test with various amplitude combinations."""
        test_cases = [
            [1, 0, 0, 0],                  # |00⟩
            [0, 1, 0, 0],                  # |01⟩
            [1/np.sqrt(2), 1/np.sqrt(2), 0, 0],  # (|00⟩ + |01⟩)/√2
            [1/2, 1/2, 1/2, 1/2],         # Equal superposition
            [1+1j, 0, 0, 1-1j],           # Complex amplitudes
        ]
        
        for amplitudes in test_cases:
            state = prepare_two_qubit_state(amplitudes)
            
            # All should be normalized
            self.assertTrue(verify_normalization(state))
            
            # All should have correct dimension
            self.assertEqual(state.size, 4)
            
            # Probabilities should sum to 1
            probs = get_measurement_probabilities(state)
            self.assertAlmostEqual(np.sum(probs), 1.0, places=10)


# ============================================================================
# MAIN: Run tests and demonstrate functionality
# ============================================================================

if __name__ == "__main__":
    # Run unit tests
    print("="*70)
    print("RUNNING UNIT TESTS")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQuantumStatePreparation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # ========================================================================
    # DEMONSTRATION: Examples of state preparation
    # ========================================================================
    
    print("\n" + "█"*70)
    print("DEMONSTRATION: Quantum State Preparation Examples")
    print("█"*70)
    
    # Example 1: Simple basis state
    print("\n" + "-"*70)
    print("Example 1: Basis state |00⟩")
    print("-"*70)
    state = prepare_two_qubit_state([1, 0, 0, 0])
    print_state_info(state, n_qubits=2, title="Two-Qubit State |00⟩")
    
    # Example 2: Superposition
    print("\n" + "-"*70)
    print("Example 2: Superposition (|00⟩ + |01⟩)/√2")
    print("-"*70)
    state = prepare_two_qubit_state([1, 1, 0, 0])
    print_state_info(state, n_qubits=2, title="Two-Qubit Superposition")
    
    # Example 3: Bell state
    print("\n" + "-"*70)
    print("Example 3: Bell state (|00⟩ + |11⟩)/√2")
    print("-"*70)
    state = prepare_two_qubit_state([1, 0, 0, 1])
    print_state_info(state, n_qubits=2, title="Bell State (Entangled)")
    
    # Example 4: Complex amplitudes
    print("\n" + "-"*70)
    print("Example 4: Complex amplitudes")
    print("-"*70)
    state = prepare_two_qubit_state([1+1j, 0, 1j, 1])
    print_state_info(state, n_qubits=2, title="Complex Amplitudes")
    
    # Example 5: Three-qubit equal superposition
    print("\n" + "-"*70)
    print("Example 5: Three-qubit equal superposition")
    print("-"*70)
    state = prepare_three_qubit_state([1]*8)
    print_state_info(state, n_qubits=3, title="Three-Qubit Equal Superposition")
    
    # Example 6: Three-qubit GHZ state
    print("\n" + "-"*70)
    print("Example 6: Three-qubit GHZ state (|000⟩ + |111⟩)/√2")
    print("-"*70)
    amps_ghz = [1, 0, 0, 0, 0, 0, 0, 1]
    state = prepare_three_qubit_state(amps_ghz)
    print_state_info(state, n_qubits=3, title="Three-Qubit GHZ State")
    
    print("\n" + "█"*70)
    print("DEMONSTRATION COMPLETE")
    print("█"*70)
