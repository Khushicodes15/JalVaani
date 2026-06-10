import pandas as pd

files = ['cgwb_water_level.csv', 'state_water_level.csv']

for file in files:
    print(f"\n{'='*50}")
    print(f"INSPECTING FILE: {file}")
    print(f"{'='*50}")
    
    try:
        df = pd.read_csv(file)
        print("\n--- SHAPE ---")
        print(df.shape)
        
        print("\n--- COLUMNS ---")
        print(df.columns.tolist())
        
        print("\n--- DTYPES ---")
        print(df.dtypes)
        
        print("\n--- ISNULL().SUM() ---")
        print(df.isnull().sum())
        
        print("\n--- HEAD(10) ---")
        print(df.head(10).to_string())
        
    except Exception as e:
        print(f"Error reading {file}: {e}")
