import pickle
import os
import json
import numpy as np
import pandas as pd
# Import Figure class to check instance type
from matplotlib.figure import Figure 
# Import base result class from arch if needed, or check type by name
try:
    from arch.univariate.base import ARCHModelResult
except ImportError:
    ARCHModelResult = None # Set to None if arch is not installed here

print("--- Exporting Pickle Data to JSON ---")

# --- Configuration ---
output_dir = 'output_figures_and_data'
pickle_filename = 'full_market_analysis_results.pkl'
json_filename = 'full_market_analysis_results_export.json' # Output JSON file name

pickle_filepath = os.path.join(output_dir, pickle_filename)
json_filepath = os.path.join(output_dir, json_filename)

# --- Cleaning Function ---
def make_json_serializable(obj):
    """
    Recursively cleans a dictionary or list to make it JSON serializable.
    Removes Matplotlib figures and complex model objects.
    Converts numpy types to standard Python types.
    """
    if isinstance(obj, dict):
        # Create a new dict to avoid modifying the original during iteration
        new_dict = {}
        for k, v in obj.items():
            # Skip keys holding problematic objects
            if isinstance(v, Figure): # Skip Matplotlib figures
                # Optionally, replace with a placeholder string
                # new_dict[k] = "<Matplotlib Figure object removed>"
                continue 
            # Skip complex arch model result objects if ARCHModelResult is available
            if ARCHModelResult and isinstance(v, ARCHModelResult):
                 # new_dict[k] = "<ARCH Model Result object removed>"
                 continue
            # Skip the raw 'model_object' key which holds the fitted model
            if k == 'model_object':
                # new_dict[k] = "<Fitted GARCH Model object removed>"
                continue

            # Recursively clean the value
            new_dict[k] = make_json_serializable(v)
        return new_dict
    elif isinstance(obj, (list, tuple)):
        # Recursively clean items in the list/tuple
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        # Convert numpy floats to Python float, handle NaN/inf
        if np.isnan(obj):
            return None # Represent NaN as null in JSON
        elif np.isinf(obj):
            return str(obj) # Represent inf as strings "+inf" or "-inf"
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        # Convert numpy arrays to lists
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, (np.bool_)):
        # Convert numpy bool to Python bool
        return bool(obj)
    elif isinstance(obj, (np.void)):
        # Handle numpy void type (often from structured arrays)
        return None # Or convert appropriately if possible
    elif isinstance(obj, (pd.Timestamp, pd.Period)):
         # Convert pandas Timestamp/Period to ISO format string
         return obj.isoformat()
    elif pd.isna(obj):
         # Handle pandas NA types
         return None
    else:
        # Return the object if it's already serializable (str, int, float, bool, None)
        return obj

# --- Load Pickle File ---
loaded_data = None
if os.path.exists(pickle_filepath):
    try:
        with open(pickle_filepath, 'rb') as f_pickle:
            loaded_data = pickle.load(f_pickle)
        print(f"Successfully loaded data from: {pickle_filepath}")
    except Exception as e:
        print(f"Error loading pickle file '{pickle_filepath}': {e}")
        loaded_data = None
else:
    print(f"Error: Pickle file not found at '{pickle_filepath}'")

# --- Clean and Export to JSON ---
if loaded_data is not None:
    print("Cleaning data for JSON serialization...")
    try:
        cleaned_data = make_json_serializable(loaded_data)
        print("Data cleaning complete.")
        
        print(f"Exporting cleaned data to JSON file: {json_filepath}")
        with open(json_filepath, 'w', encoding='utf-8') as f_json:
            # Use indent for pretty printing, ensure_ascii=False for wider character support
            json.dump(cleaned_data, f_json, indent=4, ensure_ascii=False) 
        print("Export to JSON successful.")
        print("\nYou can now copy the content of the file:")
        print(json_filepath)
        print("and paste it into your conversation with Claude.")
        
    except Exception as e:
        print(f"Error during data cleaning or JSON export: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Cannot proceed with JSON export as data was not loaded.")

# --- End of Cell ---
