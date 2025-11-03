from two_tower import TwoTowerModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from torch import optim
import torch
import joblib
import os
from db.db_helper import DBHelper
import gc



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


def setup():
    np.random.seed(42)
    num_users = 100
    num_posts = 20000

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

    POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://appuser:changeme@postgres:5432/appdb")
    db_helper = DBHelper(POSTGRES_DSN)
    db_helper.clear_post_raw()
    db_helper.clear_user_raw()
    db_helper.insert_dataframe('users_raw', sanitize_dataframe(user_df))
    db_helper.insert_dataframe('posts_raw', sanitize_dataframe(post_df))

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
        liked_count = np.random.randint(20, 50)
        sampled_count = np.random.randint(100, 200)
        liked_posts = np.random.choice(num_posts, size=liked_count, replace=False)
        sampled_posts = np.random.choice(num_posts, size=sampled_count, replace=False)
        for post_id in sampled_posts:
            interactions.append({
                'user_id': user_id,
                'post_id': post_id,
                'label': 1 if post_id in liked_posts else 0
            })
    df = pd.DataFrame(interactions)
    interactions_df = df.copy()

    # Ensure consistent types for merging
    interactions_df['user_id'] = interactions_df['user_id'].astype(int)
    interactions_df['post_id'] = interactions_df['post_id'].astype(int)
    user_df['user_id'] = user_df['user_id'].astype(int)
    post_df['post_id'] = post_df['post_id'].astype(int)
    df['user_id'] = df['user_id'].astype(int)
    df['post_id'] = df['post_id'].astype(int)

    
    db_helper.clear_interactions_raw()
    db_helper.insert_dataframe('interactions_raw', sanitize_regular_dataframe(interactions_df))


    #create train test sets
    train_df, test_df = train_test_split(df,test_size=0.2, random_state=42 )
    train_idx = train_df.index

    # Example interaction-based features for users calculated only on training interactions
    interaction_scaler = StandardScaler()
    train_interactions = train_df.groupby('user_id')['label'].agg(['sum', 'mean']).reset_index()
    train_interactions = train_interactions.astype({'user_id': int, 'sum': float, 'mean': float})
    train_interactions.rename(columns={'sum':'num_likes','mean':'like_ratio'}, inplace=True)

    # Merge these features into train and test dataframes
    train_df = train_df.merge(train_interactions, on='user_id', how='left')
    test_df = test_df.merge(train_interactions, on='user_id', how='left')

    # Fill missing values in test set with zeros
    test_df['num_likes'] = test_df['num_likes'].fillna(0).astype(float)
    test_df['like_ratio'] = test_df['like_ratio'].fillna(0).astype(float)

    user_interaction_cont_features = user_cont_features + ['num_likes','like_ratio'] 

    # Normalize the new interaction features on train set and apply to both train and test
    train_df[['num_likes','like_ratio']] = interaction_scaler.fit_transform(train_df[['num_likes','like_ratio']])
    test_df[['num_likes','like_ratio']] = interaction_scaler.transform(test_df[['num_likes','like_ratio']])

    # Merge user features into train and test dataframes
    train_df = train_df.merge(user_df, on='user_id', how='left', suffixes=('', '_user'))
    test_df = test_df.merge(user_df, on='user_id', how='left', suffixes=('', '_user'))

    # Merge post features into train and test dataframes
    train_df = train_df.merge(post_df, on='post_id', how='left', suffixes=('', '_post'))
    test_df = test_df.merge(post_df, on='post_id', how='left', suffixes=('', '_post'))

    # Enforce consistent types after merge and reset_index
    train_df['user_id'] = train_df['user_id'].astype(int)
    train_df['post_id'] = train_df['post_id'].astype(int)
    train_df['has_profile_picture'] = train_df['has_profile_picture'].astype(float)
    train_df['label'] = train_df['label'].astype(int)

    test_df['user_id'] = test_df['user_id'].astype(int)
    test_df['post_id'] = test_df['post_id'].astype(int)
    test_df['has_profile_picture'] = test_df['has_profile_picture'].astype(float)
    test_df['label'] = test_df['label'].astype(int)

    # create user and post tensors
    user_train = torch.tensor(train_df[user_interaction_cont_features + ['has_profile_picture']].to_numpy(np.float32))
    post_train = torch.tensor(train_df[post_cont_features].to_numpy(np.float32))
    label_train = torch.tensor(train_df['label'].to_numpy(np.float32))

    user_test = torch.tensor(test_df[user_interaction_cont_features + ['has_profile_picture']].to_numpy(np.float32))
    post_test = torch.tensor(test_df[post_cont_features].to_numpy(np.float32))
    label_test = torch.tensor(test_df['label'].to_numpy(np.float32))

    return user_train, post_train, label_train, user_test, post_test, label_test, user_scaler, post_scaler, df, train_idx, user_interaction_cont_features, user_df, post_df

def run(user_train, post_train, label_train, user_test, post_test, label_test, epochs=50 ):
    hidden_dims = 64
    user_dim = user_train.shape[1]
    post_dim = post_train.shape[1]
    model = TwoTowerModel(user_dim=user_dim,
                      post_dim=post_dim,
                      hidden_dim=hidden_dims,
                      dropout=0.2)
    
    ratio = (label_train==0).sum() / (label_train==1).sum()
    pos_weight = torch.tensor([min(ratio, 10.0)])
    print(f"positive weight:\t{pos_weight}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

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
    torch.cuda.empty_cache()
    gc.collect()
    return model, user_embeddings, post_embeddings, user_dim, post_dim, hidden_dims

def serialize(model, user_scaler, post_scaler, model_path="/app/models/two_tower_model.pth", 
              user_scaler_path="/app/models/user_scaler.pkl", post_scaler_path="/app/models/post_scaler.pkl"):
    

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(user_scaler, user_scaler_path)
    joblib.dump(post_scaler, post_scaler_path)

    print(f"Model saved at {model_path}")
    print(f"User scaler saved at {user_scaler_path}")
    print(f"Post scaler saved at {post_scaler_path}")




def evaluate_model(model, user_test, post_test, label_test, user_ids, top_k_list=[10, 20, 50]):
    """
    Evaluate a two-tower model for feed ranking.
    Uses raw logits for ranking to avoid threshold issues with imbalanced data.
    Computes ROC-AUC, optimal threshold, accuracy at optimal threshold, and top-k metrics.
    Top-K metrics are computed per user and averaged.
    """
    model.eval()
    with torch.no_grad():
        logits = model(user_test, post_test)         # raw logits for ranking
        probs = torch.sigmoid(logits).cpu().numpy()  # for threshold-based metrics
        logits_np = logits.cpu().numpy()            # for ranking/top-k
        labels = label_test.cpu().numpy()
        user_ids_np = np.array(user_ids)

    # ROC AUC using probabilities
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = np.nan

    # Optimal threshold by F1 (for reporting thresholded metrics)
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5
    accuracy_opt = ((probs >= optimal_threshold).astype(int) == labels).mean()

    # Prepare table for metrics
    table_rows = [
        {'metric': 'roc_auc', 'value': auc},
        {'metric': 'optimal_threshold', 'value': optimal_threshold},
        {'metric': 'accuracy_at_optimal_threshold', 'value': accuracy_opt}
    ]

    # Compute top-k metrics per user and average
    for k in top_k_list:
        precisions, recalls, ndcgs, f1s = [], [], [], []
        unique_users = np.unique(user_ids_np)
        for uid in unique_users:
            idx = np.where(user_ids_np == uid)[0]
            if len(idx) == 0:
                continue
            user_logits = logits_np[idx].flatten()
            user_labels = labels[idx].flatten()
            if len(user_labels) == 0:
                continue
            # Top-K indices for this user
            top_k_user_idx = np.argsort(-user_logits)[:min(k, len(user_logits))]
            top_k_labels = user_labels[top_k_user_idx]
            precision_at_k = top_k_labels.sum() / min(k, len(user_logits))
            recall_at_k = top_k_labels.sum() / user_labels.sum() if user_labels.sum() > 0 else 0
            discounts = 1 / np.log2(np.arange(2, min(k, len(user_logits)) + 2))
            dcg = (top_k_labels * discounts).sum()
            idcg = (np.sort(user_labels)[-min(k, len(user_logits)):][::-1] * discounts).sum()
            ndcg = dcg / idcg if idcg > 0 else 0
            f1_at_k = (2 * precision_at_k * recall_at_k) / (precision_at_k + recall_at_k + 1e-8) if (precision_at_k + recall_at_k) > 0 else 0
            precisions.append(precision_at_k)
            recalls.append(recall_at_k)
            ndcgs.append(ndcg)
            f1s.append(f1_at_k)
        # Average across users
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0
        avg_f1 = np.mean(f1s) if f1s else 0
        table_rows.append({'metric': f'Precision@{k}', 'value': avg_precision})
        table_rows.append({'metric': f'Recall@{k}', 'value': avg_recall})
        table_rows.append({'metric': f'NDCG@{k}', 'value': avg_ndcg})
        table_rows.append({'metric': f'F1@{k}', 'value': avg_f1})

    # Convert to DataFrame for printing
    metrics_df = pd.DataFrame(table_rows)
    print("\nEvaluation Metrics Summary:")
    print(metrics_df.to_string(index=False))

    return metrics_df

if __name__ == '__main__':

    user_train, post_train, label_train, user_test, post_test, label_test, user_scaler, post_scaler, df, train_idx, user_interaction_cont_features, user_df, post_df  = setup()
    model, user_embeddings, post_embeddings, user_dim, post_dim, hidden_dims = run(user_train, post_train, label_train, user_test, post_test, label_test, epochs=50)
    # Get user_ids for test set for per-user metrics
    # test_df is not in scope here, but we can reconstruct user_ids from df (the full interactions) and train_idx
    test_idx = df.index.difference(train_idx)
    user_ids_test = df.loc[test_idx, "user_id"].values if hasattr(df, "loc") else None
    if user_ids_test is None or len(user_ids_test) != len(label_test):
        # fallback: try to get user_ids from user_test shape (not ideal, but fallback)
        user_ids_test = np.zeros(len(label_test), dtype=int)
    evaluate_model(model, user_test, post_test, label_test, user_ids=user_ids_test)
    serialize(model=model, user_scaler=user_scaler, post_scaler=post_scaler)

    # insert the embeddings into the db.
    POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://appuser:changeme@postgres:5432/appdb")
    db_helper = DBHelper(POSTGRES_DSN)
    # Clear existing embeddings
    db_helper.clear_user_embeddings()
    db_helper.clear_post_embeddings()
    

    # Prepare user embeddings for batch insert (insert each unique user once)
    # Use user_interaction_cont_features to match the features used in training
    user_emb_np = model.user_tower(user_train).detach().cpu().numpy()
    user_ids_all = user_df['user_id'].values
    # Map from user_id to embedding (last occurrence in train set, but all should be same for each user)
    user_id_to_emb = {}
    for uid, emb in zip(user_ids_all, user_emb_np):
        user_id_to_emb[int(uid)] = emb.astype(np.float32)
    user_records = [(uid, emb.tolist()) for uid, emb in user_id_to_emb.items()]
    db_helper.insert_user_embeddings_batch(user_records)

    # Prepare post embeddings for batch insert (insert each unique post once)
    post_embeddings = model.post_tower(post_train)
    post_emb_np = post_embeddings.detach().cpu().numpy()
    post_ids_all = post_df['post_id'].iloc[train_idx].values
    post_id_to_emb = {}
    for pid, emb in zip(post_ids_all, post_emb_np):
        post_id_to_emb[int(pid)] = emb.astype(np.float32)
    post_records = [(pid, emb.tolist()) for pid, emb in post_id_to_emb.items()]
    db_helper.insert_post_embeddings_batch(post_records)



    model_dims_path = "/app/models/model_dims.pkl"
    model_dims = {
        "user_dim": user_dim,
        "post_dim": post_dim,
        "hidden_dims": hidden_dims 
    }

    joblib.dump(model_dims, model_dims_path)
    print(f"model dimension saved at:   {model_dims_path}")

    print("Report")
    print(f"STD of user embedding(overall): \t {np.std(user_emb_np)}")
    print(f"STD of post embedding(overall): \t {np.std(post_emb_np)}")
