from two_tower import TwoTowerModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from torch import optim
import torch
import joblib
import os
from db.db_helper import DBHelper


def setup():
    np.random.seed(42)
    num_users = 100
    num_posts = 200

# User features
    user_df = pd.DataFrame({
        'user_id': np.arange(num_users),
        'age': np.random.randint(13, 65, size=num_users),
        'gender': np.random.randint(0, 2, size=num_users),
        'num_friends': np.random.randint(50, 500, size=num_users),
        'avg_likes_received': np.random.rand(num_users) * 100,
        'avg_comments_received': np.random.rand(num_users) * 50,
        'avg_shares_received': np.random.rand(num_users) * 30,
        'active_days_last_week': np.random.randint(0, 7, size=num_users),
        'time_spent_last_week': np.random.rand(num_users) * 1000,
        'num_groups': np.random.randint(1, 20, size=num_users),
        'has_profile_picture': np.random.randint(0, 2, size=num_users)
    })

    # Post features
    post_df = pd.DataFrame({
        'post_id': np.arange(num_posts),
        'post_length': np.random.randint(20, 2000, size=num_posts),
        'num_images': np.random.randint(0, 5, size=num_posts),
        'num_videos': np.random.randint(0, 3, size=num_posts),
        'num_hashtags': np.random.randint(0, 10, size=num_posts),
        'author_followers': np.random.randint(100, 10000, size=num_posts),
        'author_following': np.random.randint(50, 5000, size=num_posts),
        'author_posts_last_week': np.random.randint(0, 20, size=num_posts),
        'post_type': np.random.choice(['text', 'image', 'video'], size=num_posts),
        'post_time_hour': np.random.randint(0,24,size=num_posts),
        'is_boosted': np.random.randint(0,2,size=num_posts)
    })
    ### 3️⃣ Feature Engineering

    # - 3a: Normalize continuous features
    # - 3b: One-hot encode categorical features
    user_cont_features = ['age','num_friends','avg_likes_received','avg_comments_received',
                        'avg_shares_received','active_days_last_week','time_spent_last_week','num_groups']
    user_scaler = StandardScaler()
    user_df[user_cont_features] = user_scaler.fit_transform(user_df[user_cont_features])
    user_df['has_profile_picture'] = user_df['has_profile_picture'].astype(float)

    post_scaler = StandardScaler()
    post_cont_features = ['post_length','num_images','num_videos','num_hashtags',
                        'author_followers','author_following','author_posts_last_week']
    post_df[post_cont_features] = post_scaler.fit_transform(post_df[post_cont_features])
    post_df = pd.get_dummies(post_df, columns=['post_type','post_time_hour'])

    interactions = []
    for user_id in range(num_users):
        liked_posts = np.random.choice(num_posts, size=20, replace=False)
        for post_id in range(num_posts):
            interactions.append({
                'user_id': user_id,
                'post_id': post_id,
                'label': 1 if post_id in liked_posts else 0
            })
    df = pd.DataFrame(interactions)
    interactions_df = df.copy()
    df = df.merge(user_df, on='user_id', how='left')
    df = df.merge(post_df, on='post_id', how='left')
    df = df.reset_index(drop=True)

    # create user and post tensors
    users = torch.tensor(df[user_cont_features + ['has_profile_picture']].to_numpy(np.float32))
    posts = torch.tensor(df[post_cont_features].to_numpy(np.float32))
    labels = torch.tensor(df['label'].to_numpy(np.float32))


    #create train test sets
    train_idx, test_idx = train_test_split(range(len(labels)),test_size=0.2, random_state=42 )

    user_train = users[train_idx]
    user_test = users[test_idx]

    post_train = posts[train_idx]
    post_test = posts[test_idx]

    label_train = labels[train_idx]
    label_test = labels[test_idx]

    return user_train, post_train, label_train, user_test, post_test, label_test, user_scaler, post_scaler, user_df, post_df,interactions_df 

def run(user_train, post_train, label_train, user_test, post_test, label_test, epochs=50 ):
    hidden_dims = 64
    user_dim = user_train.shape[1]
    post_dim = post_train.shape[1]
    model = TwoTowerModel(user_dim=user_dim,
                      post_dim=post_dim,
                      hidden_dim=hidden_dims,
                      dropout=0.2)
    
    pos_weight = torch.tensor([(label_train==0).sum() / (label_train==1).sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(user_train, post_train)
        loss = criterion(logits, label_train)
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(user_test, post_test)
            test_loss = criterion(test_logits, label_test)
            test_probs = torch.sigmoid(test_logits)  # probabilities for inspection
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: Train loss = {loss.item():.4f}, Test loss = {test_loss.item():.4f}")
    with torch.no_grad():
        user_embeddings = model.user_tower(user_train)
        post_embeddings = model.post_tower(post_train)

    return model, user_embeddings, post_embeddings, user_dim, post_dim

def serialize(model, user_scaler, post_scaler, model_path="/app/models/two_tower_model.pth", 
              user_scaler_path="/app/models/user_scaler.pkl", post_scaler_path="/app/models/post_scaler.pkl"):
    

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(user_scaler, user_scaler_path)
    joblib.dump(post_scaler, post_scaler_path)

    print(f"Model saved at {model_path}")
    print(f"User scaler saved at {user_scaler_path}")
    print(f"Post scaler saved at {post_scaler_path}")

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
    """
    for col in df.columns:
        df[col] = df[col].apply(lambda x: int(x) if isinstance(x, (bool, np.bool_, np.integer)) 
                                                else float(x) if isinstance(x, (np.floating,)) 
                                                else str(x))
    return df

if __name__ == '__main__':
    user_train, post_train, label_train, user_test, post_test, label_test, user_scaler, post_scaler, user_df, post_df,interactions_df  = setup()
    model, user_embeddings, post_embeddings, user_dim, post_dim = run(user_train, post_train, label_train, user_test, post_test, label_test)
    serialize(model=model, user_scaler=user_scaler, post_scaler=post_scaler)

    # insert the embeddings into the db.
    POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://appuser:changeme@postgres:5432/appdb")
    db_helper = DBHelper(POSTGRES_DSN)
    # Clear existing embeddings
    db_helper.clear_user_embeddings()
    db_helper.clear_post_embeddings()
    db_helper.clear_post_raw()
    db_helper.clear_user_raw()
    db_helper.clear_interactions_raw()

    # Prepare user embeddings for batch insert
    user_emb_np = user_embeddings.detach().cpu().numpy()
    user_ids = np.arange(user_emb_np.shape[0])
    user_records = [(int(uid), emb.astype(np.float32).tolist()) for uid, emb in zip(user_ids, user_emb_np)]
    db_helper.insert_user_embeddings_batch(user_records)
    # Prepare post embeddings for batch insert
    post_emb_np = post_embeddings.detach().cpu().numpy()
    post_ids = np.arange(post_emb_np.shape[0])
    post_records = [(int(pid), emb.astype(np.float32).tolist()) for pid, emb in zip(post_ids, post_emb_np)]
    db_helper.insert_post_embeddings_batch(post_records)

    db_helper.insert_dataframe('users_raw', sanitize_dataframe(user_df))
    db_helper.insert_dataframe('posts_raw', sanitize_dataframe(post_df))
    db_helper.insert_dataframe('interactions_raw', sanitize_regular_dataframe(interactions_df))

    model_dims_path = "/app/models/model_dims.pkl"
    model_dims = {
        "user_dim": user_dim,
        "post_dim": post_dim 
    }

    joblib.dump(model_dims, model_dims_path)
    print(f"model dimension saved at:   {model_dims_path}")