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
    print("\n========================================================")
    print("🚀 INIT: Deep Learning Recommendation Pipeline (DLRM)")
    print("========================================================")

    # 1️⃣ Generate Synthetic data
    print("\n[Stage 1.5] Synthesizing Data (Simulating Lambda Architecture)")
    print("   => Distinguishing Dense context features from Sparse identity markers...")
    data = DataGenerator()
    data.create()
    print("   => Synthesized 100 Users and 20,000 Posts.")

    # 2️⃣ Train the Two tower model
    print("\n[Stage 1] Training Two-Tower Model (Retrieval Stage)")
    print("   => Objective: Map User and Post features into a geometric 64D space to defeat the Cold-Start problem.")
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
    print("\n[Stage 3] Training Multi-Task DLRM Ranker (Precision Stage)")
    print("   => Objective: Hashing Sparse IDs and computing Dot-Product Interactions with Dense Bottom MLP outputs.")
    ranker = Ranker(data, trainer)
    ranker.initialize()
    ranker.train(epoch=50, lr=0.05)
    print("\n   => Executing Re-Ranking Calibration Metrics (ROC-AUC / NDCG)...")
    ranker.evaluate_model()

    #  4️⃣ Insert the embeddings into the db.
    print("\n[System Sync] Pushing Dense Embeddings to PostgreSQL pgvector")
    print("   => Objective: Making vectors available for K-Means Clustering (IVF) and FAISS Retrieval.")
    POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://appuser:changeme@postgres:5432/appdb")
    db_helper = DBHelper(POSTGRES_DSN)
    # Clear existing embeddings
    db_helper.clear_user_embeddings()
    db_helper.clear_post_embeddings()
    db_helper.insert_user_embeddings_batch(user_records)
    db_helper.insert_post_embeddings_batch(post_records)
    print("   => Vectors injected successfully.")
    
    print("\n[System Flush] Serializing pipeline components to disk for deployment...")
    data.serialize()
    trainer.serialize()
    ranker.serialize()


    model_dims_path = "./models/model_dims.pkl"
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