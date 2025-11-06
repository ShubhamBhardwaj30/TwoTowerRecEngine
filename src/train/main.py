import numpy as np
from two_tower_trainer import TwoTowerTrainer
from db.db_helper import DBHelper
import os
import joblib
from data_gen import DataGenerator
from ranker_trainer import Ranker
import torch

def main():
    data = DataGenerator()
    trainer = TwoTowerTrainer(data)
    trainer.setup()
    trainer.train(epochs=50, lr=0.05)
    # Get user_ids for test set for per-user metrics
    test_idx = data.df.index.difference(data.train_idx)
    user_ids_test = data.df.loc[test_idx, "user_id"].values if hasattr(data.df, "loc") else None
    if user_ids_test is None or len(user_ids_test) != len(data.tower_label_test):
        user_ids_test = np.zeros(len(data.tower_label_test), dtype=int)
    trainer.evaluate_model()
    trainer.serialize()


   

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

    
    # 2. Prepare training input for ranker
    ranker = Ranker(data)
    ranker.train()

     # insert the embeddings into the db.
    POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://appuser:changeme@postgres:5432/appdb")
    db_helper = DBHelper(POSTGRES_DSN)
    # Clear existing embeddings
    db_helper.clear_user_embeddings()
    db_helper.clear_post_embeddings()
    db_helper.insert_user_embeddings_batch(user_records)
    db_helper.insert_post_embeddings_batch(post_records)



    model_dims_path = "/app/models/model_dims.pkl"
    model_dims = {
        "user_dim": trainer.user_dim,
        "post_dim": trainer.post_dim,
        "hidden_dims": trainer.hidden_dims 
    }

    joblib.dump(model_dims, model_dims_path)
    print(f"model dimension saved at:   {model_dims_path}")

    print("Report")
    print(f"STD of user embedding(overall): \t {np.std(user_emb_np)}")
    print(f"STD of post embedding(overall): \t {np.std(post_emb_np)}")




if __name__ == '__main__':
    main()