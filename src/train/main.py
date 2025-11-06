import numpy as np
from two_tower_trainer import TwoTowerTrainer
from db.db_helper import DBHelper
import os
import joblib
from data_gen import DataGenerator
from ranker_trainer import Ranker
import torch
from utils import prepare_embeddings

def main():
    
    # 1️⃣ Generate Synthetic data
    data = DataGenerator()
    data.create()

    # 2️⃣ Train the Two tower model
    trainer = TwoTowerTrainer(data)
    trainer.initialize()
    trainer.train(epochs=50, lr=0.05)
    # Get user_ids for test set for per-user metrics
    test_idx = data.df.index.difference(data.train_idx)
    user_ids_test = data.df.loc[test_idx, "user_id"].values if hasattr(data.df, "loc") else None
    if user_ids_test is None or len(user_ids_test) != len(data.tower_label_test):
        user_ids_test = np.zeros(len(data.tower_label_test), dtype=int)
    trainer.evaluate_model()
    

    # 3️⃣ Prepare embeddings for DB insert
    user_records, post_records, user_emb_np, post_emb_np = prepare_embeddings(trainer=trainer, data=data)
    
    # 3️⃣ Train the ranking model
    ranker = Ranker(data, trainer)
    ranker.initialize()
    ranker.train(epoch=50, lr=0.05)
    ranker.evaluate_model()

    #  4️⃣ Insert the embeddings into the db.
    POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://appuser:changeme@postgres:5432/appdb")
    db_helper = DBHelper(POSTGRES_DSN)
    # Clear existing embeddings
    db_helper.clear_user_embeddings()
    db_helper.clear_post_embeddings()
    db_helper.insert_user_embeddings_batch(user_records)
    db_helper.insert_post_embeddings_batch(post_records)
    
    data.serialize()
    trainer.serialize()
    ranker.serialize()


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