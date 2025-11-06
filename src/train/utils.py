import numpy as np
import pandas as pd
import os
import joblib
import torch

def sanitize_dataframe(df: pd.DataFrame):
    """
    Ensure all DataFrame columns have native Python types compatible with psycopg2:
    - bool → int
    - NumPy integers → int
    - NumPy floats → float
    - Others → str
    """
    for col in df.columns:
        # Booleans
        if df[col].dtype == 'bool' or df[col].dtype.name == 'boolean':
            df[col] = df[col].astype(int).values.tolist()
        # NumPy integers
        elif np.issubdtype(df[col].dtype, np.integer):
            df[col] = [int(x) for x in df[col].to_numpy()]
        # NumPy floats
        elif np.issubdtype(df[col].dtype, np.floating):
            df[col] = [float(x) for x in df[col].to_numpy()]
        # Objects/strings
        else:
            df[col] = df[col].astype(str).values.tolist()
    return df

def sanitize_regular_dataframe(df: pd.DataFrame):
    """
    Robustly convert all boolean and NumPy scalar types to native Python types
    for psycopg2 compatibility without affecting the rest of the code.
    Ensure 'label' column remains numeric.
    """
    for col in df.columns:
        if col == 'label':
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].apply(lambda x: int(x) if isinstance(x, (bool, np.bool_, np.integer))
                                                    else float(x) if isinstance(x, (np.floating,))
                                                    else str(x))
    return df
