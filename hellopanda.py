try:
    import pandas as pd
    print("Pandas is installed.")
    print(f"Pandas version: {pd.__version__}")
except ImportError:
    print("Pandas is not installed.")
    print("To install Pandas, run: pip install pandas")