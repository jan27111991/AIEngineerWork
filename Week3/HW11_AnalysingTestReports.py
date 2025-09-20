import numpy as np

class TestReport:
    def __init__(self, execution_times):
        self.execution_times = execution_times

    def average_time(self):
        return np.mean(self.execution_times)

    def max_time(self):
        return np.max(self.execution_times)

class RegressionReport(TestReport):
    def __init__(self, execution_times):
        super().__init__(execution_times)

    def slow_tests(self, threshold):
        return self.execution_times[self.execution_times > threshold]


if __name__ == "__main__":
    # Create a NumPy array with 10 execution times
    times = np.array([12, 18, 25, 30, 22, 17, 29, 35, 40, 15])
    # Create RegressionReport object
    report = RegressionReport(times)
    # Display average, max, and slow tests
    print("Average execution time:", report.average_time())
    print("Maximum execution time:", report.max_time())
    threshold = 25
    print(f"Tests slower than {threshold} seconds:", report.slow_tests(threshold))