from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
import redis
import json
import torch
import joblib
from db.db_helper import DBHelper
from api.two_tower import TwoTowerModel
from api.data_model import UpsertItem, QueryRequest

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://appuser:changeme@postgres:5432/appdb")

# Initialize Postgres DB helper
db_helper = DBHelper(POSTGRES_DSN)

# Redis client
r = redis.from_url(REDIS_URL)

app = FastAPI(title="Feed Ranking")

# Global placeholders for model and scalers
global model, user_scaler, post_scaler, model_dims
model = user_scaler = post_scaler = model_dims = None

def load_model():
    global model, user_scaler, post_scaler, model_dims
    model_path = "/app/models/two_tower_model.pth"
    user_scaler_path = "/app/models/user_scaler.pkl"
    post_scaler_path = "/app/models/post_scaler.pkl"
    model_dims_path = "/app/models/model_dims.pkl" 

    user_scaler = joblib.load(user_scaler_path)
    post_scaler = joblib.load(post_scaler_path)
    model_dims = joblib.load(model_dims_path)

    user_dim = model_dims['user_dim']
    post_dim = model_dims['post_dim']
    hidden_dim = model_dims['hidden_dims']

    model = TwoTowerModel(user_dim=user_dim, post_dim=post_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("Model loaded")


@app.on_event("startup")
def start():
    load_model()


@app.post("/upsert")
def upsert_item(item: UpsertItem):
    """
    Convert incoming post features to the same feature vector used at training,
    scale the continuous features with the training scaler, produce a post embedding
    via the post_tower, and insert the raw embedding into the DB (do NOT scale
    the embedding itself).
    """
    # 1) Continuous features in the same order as training
    post_cont = np.array([
        item.post_length,
        item.num_images,
        item.num_videos,
        item.num_hashtags,
        item.author_followers,
        item.author_following,
        item.author_posts_last_week
    ], dtype=np.float32).reshape(1, -1)  # shape (1, 7)

    # 2) Scale continuous features using the training post_scaler
    try:
        post_cont_scaled = post_scaler.transform(post_cont).reshape(-1)  # shape (7,)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"post_scaler.transform failed: {e}")

    # 3) One-hot encode post_type (text, image, video) — same order used during training
    post_type_ohe = np.array([
        1 if item.post_type == 'text' else 0,
        1 if item.post_type == 'image' else 0,
        1 if item.post_type == 'video' else 0
    ], dtype=np.float32)

    # 4) One-hot encode post_time_hour (0..23)
    post_hour_ohe = np.array([1 if item.post_time_hour == i else 0 for i in range(24)], dtype=np.float32)

    # 5) is_boosted (binary)
    is_boosted = np.array([item.is_boosted], dtype=np.float32)

    # 6) Final feature vector concatenation must match training post_feature ordering
    post_features = np.concatenate([post_cont_scaled, post_type_ohe, post_hour_ohe, is_boosted]).astype(np.float32)

    # Sanity check: make sure the assembled feature vector length equals post_dim
    if 'post_dim' in globals():
        expected = post_dim
    elif model is not None and hasattr(model, "post_tower"):
        # derive from loaded model if post_dim not in globals
        expected = model.post_tower[0].in_features if hasattr(model.post_tower[0], "in_features") else None
    else:
        expected = None

    if expected is not None and post_features.shape[0] != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Post feature dimension mismatch: expected {expected}, got {post_features.shape[0]}"
        )

    post_input = torch.tensor(post_features).unsqueeze(0)  # shape (1, post_dim)

    # 7) Produce embedding using the post_tower (do NOT scale/transform the resulting embedding)
    with torch.no_grad():
        post_emb = model.post_tower(post_input).detach().cpu().numpy()[0].astype(np.float32)

    # Optional: if you use cosine similarity in DB, you may want to L2-normalize embeddings:
    # norm = np.linalg.norm(post_emb)
    # if norm > 0:
    #     post_emb = (post_emb / norm).astype(np.float32)

    # 8) Insert raw embedding into DB
    db_helper.insert_post_embedding(item.post_id, post_emb.tolist())
    return {"status": "ok", "post_id": item.post_id}

@app.post("/query")
def query(q: QueryRequest):
    cache_key = f"query:{q.top_k}:{q.age}:{q.gender}:{q.num_friends}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    with torch.no_grad():
        # 1. Raw continuous features (only 8 base features)
        user_base_features = np.array([
            q.age, q.num_friends, q.avg_likes_received, q.avg_comments_received,
            q.avg_shares_received, q.active_days_last_week, q.time_spent_last_week,
            q.num_groups
        ]).reshape(1, -1)  # shape (1,8)

        # 2. Scale base features
        user_base_scaled = user_scaler.transform(user_base_features)  # shape (1,8)

        # 3. Append num_likes and like_ratio as 0.0, and has_profile_picture
        user_input_np = np.hstack([user_base_scaled, [[0.0, 0.0]], [[q.has_profile_picture]]])  # shape (1,11)

        # 4. Convert to tensor
        user_input = torch.tensor(user_input_np, dtype=torch.float32)

        # 5. Feed into model
        user_emb = model.user_tower(user_input).detach().numpy()[0]
            

    results = db_helper.query_similar_posts_ann(user_emb.tolist(), top_k=q.top_k)
    # print(results)
    hits = [{"post_id": int(r[0]), "similarity_distance": round(float(r[1]), 4)} for r in results]

    out = {"hits": hits}
    r.set(cache_key, json.dumps(out), ex=60)
    return out

@app.get("/reload")
def reload():
    load_model()