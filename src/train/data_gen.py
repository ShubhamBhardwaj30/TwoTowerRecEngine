import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from db.db_helper import DBHelper
from utils import sanitize_dataframe, sanitize_regular_dataframe
import joblib

class DataGenerator:

    def __init__(self):
        self.user_train = None
        self.post_train = None
        self.tower_label_train = None
        self.mhead_label_train = None
        self.user_test = None
        self.post_test = None
        self.tower_label_test = None
        self.mhead_label_test = None
        self.user_scaler = None
        self.post_scaler = None
        self.df = None
        self.train_idx = None
        self.user_df = None
        self.post_df = None
        self.train_df = None
        self.test_df = None
        self.user_scaler = StandardScaler()
        self.post_scaler = StandardScaler()
    
    def create(self):
        np.random.seed(42)
        num_users = 100
        num_posts = 20000

        # User features
        self.user_df = pd.DataFrame({
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
        self.post_df = pd.DataFrame({
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
        db_helper.insert_dataframe('users_raw', sanitize_dataframe(self.user_df))
        db_helper.insert_dataframe('posts_raw', sanitize_dataframe(self.post_df))

    ### 3️⃣ Feature Engineering

    # - 3a: Normalize continuous features
    # - 3b: One-hot encode categorical features
        user_cont_features = ['age','num_friends','avg_likes_received','avg_comments_received',
                            'avg_shares_received','active_days_last_week','time_spent_last_week','num_groups']
        
        self.user_df[user_cont_features] = self.user_scaler.fit_transform(self.user_df[user_cont_features])
        self.user_df['has_profile_picture'] = self.user_df['has_profile_picture'].astype(float)

        post_cont_features = ['post_length','num_images','num_videos','num_hashtags',
                            'author_followers','author_following','author_posts_last_week']
        self.post_df[post_cont_features] = self.post_scaler.fit_transform(self.post_df[post_cont_features])
        self.post_df = pd.get_dummies(self.post_df, columns=['post_type','post_time_hour'])

        all_records = []

        for user_id in range(num_users):
            # 1️⃣ Sample liked posts
            liked_count = np.random.randint(20, 50)
            sampled_posts_liked = np.random.choice(num_posts, size=np.random.randint(100, 200), replace=False)
            liked_posts = np.random.choice(sampled_posts_liked, size=liked_count, replace=False)

            # 2️⃣ Sample commented posts
            commented_count = np.random.randint(5, 10)
            sampled_posts_commented = np.random.choice(num_posts, size=np.random.randint(100, 200), replace=False)
            commented_posts = np.random.choice(sampled_posts_commented, size=commented_count, replace=False)

            # 3️⃣ Sample shared posts
            shared_count = np.random.randint(10, 30)
            sampled_posts_shared = np.random.choice(num_posts, size=np.random.randint(100, 200), replace=False)
            shared_posts = np.random.choice(sampled_posts_shared, size=shared_count, replace=False)

            # 4️⃣ Combine all posts interacted with
            all_interacted_posts = np.unique(np.concatenate([sampled_posts_liked, sampled_posts_commented, sampled_posts_shared]))

            for post_id in all_interacted_posts:
                all_records.append({
                    'user_id': user_id,
                    'post_id': post_id,
                    'liked': float(post_id in liked_posts),
                    'commented': float(post_id in commented_posts),
                    'shared': float(post_id in shared_posts)
                })

        # 5️⃣ Create final DataFrame at once
        self.df = pd.DataFrame(all_records, columns=['user_id', 'post_id', 'liked', 'commented', 'shared']).astype({
            'user_id': 'int64',
            'post_id': 'int64',
            'liked': 'float32',
            'commented': 'float32',
            'shared': 'float32'
        })
        interactions_df = self.df.copy()

        # Ensure consistent types for merging
        interactions_df['user_id'] = interactions_df['user_id'].astype(int)
        interactions_df['post_id'] = interactions_df['post_id'].astype(int)
        self.user_df['user_id'] = self.user_df['user_id'].astype(int)
        self.post_df['post_id'] = self.post_df['post_id'].astype(int)
        self.df['user_id'] = self.df['user_id'].astype(int)
        self.df['post_id'] = self.df['post_id'].astype(int)

        db_helper.clear_interactions_raw_v2()
        db_helper.insert_dataframe('interactions_raw_v2', sanitize_dataframe(interactions_df))

        # combine labels for the two tower model
        self.df['label'] = ((self.df['liked'] == 1) | (self.df['commented'] == 1) | (self.df['shared'] == 1)).astype(int)

        #create train test sets
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_idx = self.train_df.index

        # Merge user features into train and test dataframes
        self.train_df = self.train_df.merge(self.user_df, on='user_id', how='left', suffixes=('', '_user'))
        self.test_df = self.test_df.merge(self.user_df, on='user_id', how='left', suffixes=('', '_user'))

        # Merge post features into train and test dataframes
        self.train_df = self.train_df.merge(self.post_df, on='post_id', how='left', suffixes=('', '_post'))
        self.test_df = self.test_df.merge(self.post_df, on='post_id', how='left', suffixes=('', '_post'))

        # Label features
        label_cols = ['liked', 'commented', 'shared']

        # Enforce consistent types after merge and reset_index
        self.train_df['user_id'] = self.train_df['user_id'].astype(int)
        self.train_df['post_id'] = self.train_df['post_id'].astype(int)
        self.train_df['has_profile_picture'] = self.train_df['has_profile_picture'].astype(float)
        self.train_df[label_cols] = self.train_df[label_cols].astype(int)
        self.train_df['label'] = self.train_df['label'].astype(int)
        self.test_df['user_id'] = self.test_df['user_id'].astype(int)
        self.test_df['post_id'] = self.test_df['post_id'].astype(int)
        self.test_df['has_profile_picture'] = self.test_df['has_profile_picture'].astype(float)
        self.test_df['label'] = self.test_df['label'].astype(int)

        # create user and post tensors
        self.user_train = torch.tensor(self.train_df[user_cont_features + ['has_profile_picture']].to_numpy(dtype=np.float32))
        self.post_train = torch.tensor(self.train_df[post_cont_features].to_numpy(dtype=np.float32))
        self.tower_label_train = torch.tensor(self.train_df['label'].to_numpy(dtype=np.float32))
        self.mhead_label_train = torch.tensor(self.train_df[label_cols].to_numpy(dtype=np.float32))

        self.user_test = torch.tensor(self.test_df[user_cont_features + ['has_profile_picture']].to_numpy(dtype=np.float32))
        self.post_test = torch.tensor(self.test_df[post_cont_features].to_numpy(dtype=np.float32))
        self.tower_label_test = torch.tensor(self.test_df['label'].to_numpy(dtype=np.float32))
        self.mhead_label_test = torch.tensor(self.test_df[label_cols].to_numpy(dtype=np.float32))

    def serialize(self, model_path="/app/models/",
                  user_scaler_path="/app/models/user_scaler.pkl", post_scaler_path="/app/models/post_scaler.pkl"):
        """
        Serialize the model and scalers to disk. Uses self.model, self.user_scaler, self.post_scaler.
        """
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        joblib.dump(self.user_scaler, user_scaler_path)
        joblib.dump(self.post_scaler, post_scaler_path)
        print(f"User scaler saved at {user_scaler_path}")
        print(f"Post scaler saved at {post_scaler_path}")