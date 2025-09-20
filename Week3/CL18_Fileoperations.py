# Write Operation
with open("report.txt", "w") as file:
    file.write("TestCase1 - Passed\n")
    file.write("TestCase2 - Failed\n")
    file.write("TestCase3 - Passed\n")

# Append Operation
with open("report.txt", "a") as file:
    file.write("TestCase4 - Passed\n")
    file.write("TestCase5 - Failed\n")

# Read Operation and Summary
total_tests = 0
passed_count = 0
failed_count = 0

with open("report.txt", "r") as file:
    print("Test Results:")
    for line in file:
        line = line.strip()
        print(line)
        total_tests += 1
        if "Passed" in line:
            passed_count += 1
        elif "Failed" in line:
            failed_count += 1

print("\nSummary:")
print(f"Total Tests: {total_tests}")
print(f"Passed: {passed_count}")
print(f"Failed: {failed_count}")