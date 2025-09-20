import numpy as np

class ManualTester:
    def analyze(self, data):
        print("ManualTester - First 5 test execution times:", data[:5])

class AutomationTester:
    def analyze(self, data):
        print("AutomationTester - Fastest test case:", np.min(data))

class PerformanceTester:
    def analyze(self, data):
        print("PerformanceTester - 95th percentile execution time:", np.percentile(data, 95))

def show_analysis(tester, data):
    tester.analyze(data)

if __name__ == "__main__":
    # Create a NumPy array with at least 12 execution times
    times = np.array([12, 18, 25, 30, 22, 17, 29, 35, 40, 15, 28, 33])
    # Create objects for each tester role
    manual = ManualTester()
    automation = AutomationTester()
    performance = PerformanceTester()
    # Call show_analysis for each tester
    show_analysis(manual, times)
    show_analysis(automation, times)
    show_analysis(performance, times)