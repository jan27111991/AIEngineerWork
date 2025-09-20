import numpy as np

# a) Generate synthetic data: 5 cycles × 50 tests, random times between 5 and 50
np.random.seed(42)  # For reproducibility
data = np.random.randint(5, 51, size=(5, 50))
print("Synthetic Data (5x50):\n", data)

# 1. Statistical Analysis
# Average execution time per cycle
avg_per_cycle = np.mean(data, axis=1)
print("\nAverage execution time per cycle:", avg_per_cycle)

# Test case with maximum execution time in the entire dataset
max_time = np.max(data)
max_cycle, max_test = np.unravel_index(np.argmax(data), data.shape)
print(f"\nMaximum execution time: {max_time} (Cycle {max_cycle+1}, Test {max_test+1})")

# Standard deviation per cycle
std_per_cycle = np.std(data, axis=1)
print("\nStandard deviation per cycle:", std_per_cycle)

# 2. Slicing Operations
first10_cycle1 = data[0, :10]
print("\nFirst 10 test execution times from Cycle 1:", first10_cycle1)

last5_cycle5 = data[4, -5:]
print("Last 5 test execution times from Cycle 5:", last5_cycle5)

alternate_cycle3 = data[2, ::2]
print("Every alternate test from Cycle 3:", alternate_cycle3)

# 3. Arithmetic Operations
add_1_2 = data[0] + data[1]
sub_1_2 = data[0] - data[1]
print("\nElement-wise addition (Cycle 1 + Cycle 2):", add_1_2)
print("Element-wise subtraction (Cycle 1 - Cycle 2):", sub_1_2)

mul_4_5 = data[3] * data[4]
div_4_5 = data[3] / data[4]
print("Element-wise multiplication (Cycle 4 * Cycle 5):", mul_4_5)
print("Element-wise division (Cycle 4 / Cycle 5):", div_4_5)

# 4. Power Functions
squared = np.power(data, 2)
cubed = np.power(data, 3)
sqrted = np.sqrt(data)
logged = np.log(data + 1)
print("\nSquared execution times (first row):", squared[0, :5])
print("Cubed execution times (first row):", cubed[0, :5])
print("Square root (first row):", sqrted[0, :5])
print("Log-transformed (first row):", logged[0, :5])

# 5. Copy Operations
shallow = data.view()
shallow[0, 0] = 99
print("\nAfter modifying shallow copy, data[0,0]:", data[0, 0])  # Should reflect change

deep = data.copy()
deep[1, 0] = 77
print("After modifying deep copy, data[1,0]:", data[1, 0])  # Should NOT reflect change

# 6. Filtering with Conditions
cycle2_over30 = data[1, data[1] > 30]
print("\nCycle 2 tests > 30 seconds:", cycle2_over30)

# Tests that consistently take >25s in every cycle (column-wise all >25)
consistently_over25 = np.where(np.all(data > 25, axis=0))[0]
print("Test indices consistently >25s in every cycle:", consistently_over25 + 1)

# Replace all execution times below 10s with 10 (minimum thresholding)
thresholded = data.copy()
thresholded[thresholded < 10] = 10
print("\nData after minimum thresholding (first row):", thresholded[0, :10])