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
        orig = df[col].dtype
        # Force the column to accept plain python objects so pandas doesn't implicitly cast back to numpy
        df[col] = df[col].astype(object)
        
        # Booleans
        if orig == 'bool' or getattr(orig, 'name', '') == 'boolean':
            df[col] = [int(x) for x in df[col].to_numpy()]
        # NumPy integers
        elif np.issubdtype(orig, np.integer):
            df[col] = [int(x) for x in df[col].to_numpy()]
        # NumPy floats
        elif np.issubdtype(orig, np.floating):
            df[col] = [float(x) for x in df[col].to_numpy()]
        # Objects/strings
        else:
            df[col] = [str(x) for x in df[col].to_numpy()]
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

def prepare_embeddings(trainer, data):
        # Prepare user embeddings for batch insert (insert each unique user once)
        # Use user_interaction_cont_features to match the features used in training
        user_emb_np = trainer.get_user_embeddings()
        user_ids_all = data.user_df['user_id'].values
        # Map from user_id to embedding (last occurrence in train set, but all should be same for each user)
        user_id_to_emb = {}
        for uid, emb in zip(user_ids_all, user_emb_np):
            user_id_to_emb[int(uid)] = emb.astype(np.float32)
        user_records = [(uid, emb.tolist()) for uid, emb in user_id_to_emb.items()]

        # Prepare post embeddings for batch insert (insert each unique post once)
        post_emb_np = trainer.get_post_embeddings()
        
        post_ids_all = data.df.loc[data.train_idx, 'post_id'].unique()
        post_id_to_emb = {}
        for pid, emb in zip(post_ids_all, post_emb_np):
            post_id_to_emb[int(pid)] = emb.astype(np.float32)
        post_records = [(pid, emb.tolist()) for pid, emb in post_id_to_emb.items()]
        
        return user_records, post_records, user_emb_np, post_emb_np
