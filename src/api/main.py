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

@app.on_event("startup")
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
    hidden_dim = 64

    model = TwoTowerModel(user_dim=user_dim, post_dim=post_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("Model loaded")


@app.post("/upsert")
def upsert_item(item: UpsertItem):
    with torch.no_grad():
        # Convert UpsertItem fields to feature vector matching training post_df
        post_features = np.array([
            item.post_length,
            item.num_images,
            item.num_videos,
            item.num_hashtags,
            item.author_followers,
            item.author_following,
            item.author_posts_last_week,
            # One-hot encode post_type
            1 if item.post_type == 'text' else 0,
            1 if item.post_type == 'image' else 0,
            1 if item.post_type == 'video' else 0,
            # One-hot encode post_time_hour
            *[1 if item.post_time_hour == i else 0 for i in range(24)],
            item.is_boosted
        ], dtype=np.float32)
        post_input = torch.tensor(post_features).unsqueeze(0)
        
        post_emb = model.post_tower(post_input).detach().numpy()[0]
        post_emb = post_scaler.transform(post_emb.reshape(1, -1))[0].astype(np.float32)

    db_helper.insert_post_embedding(item.post_id, post_emb.tolist())
    return {"status": "ok", "post_id": item.post_id}


@app.post("/query")
def query(q: QueryRequest):
    cache_key = f"query:{q.top_k}:{q.age}:{q.gender}:{q.num_friends}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    with torch.no_grad():
        # 1. Raw continuous features
        user_cont = np.array([
            q.age, q.num_friends, q.avg_likes_received, q.avg_comments_received,
            q.avg_shares_received, q.active_days_last_week, q.time_spent_last_week,
            q.num_groups
        ]).reshape(1, -1)  # shape (1,8)

        # 2. Scale continuous features
        user_cont_scaled = user_scaler.transform(user_cont)  # shape (1,8)

        # 3. Append has_profile_picture as 2D
        has_profile = np.array([[q.has_profile_picture]])  # shape (1,1)
        user_input_np = np.hstack([user_cont_scaled, has_profile])  # shape (1,9)

        # 4. Convert to tensor
        user_input = torch.tensor(user_input_np, dtype=torch.float32)

        # 5. Feed into model
        user_emb = model.user_tower(user_input).detach().numpy()[0]
            

    results = db_helper.query_similar_posts(user_emb.tolist(), top_k=q.top_k)
    hits = [{"post_id": int(r[0]), "score": float(r[1])} for r in results]

    out = {"hits": hits}
    r.set(cache_key, json.dumps(out), ex=60)
    return out