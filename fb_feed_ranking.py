### 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split


### 2️⃣ Simulate Users and Posts
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
scaler = StandardScaler()
user_df[user_cont_features] = scaler.fit_transform(user_df[user_cont_features])
user_df['has_profile_picture'] = user_df['has_profile_picture'].astype(float)

post_cont_features = ['post_length','num_images','num_videos','num_hashtags',
                      'author_followers','author_following','author_posts_last_week']
post_df[post_cont_features] = scaler.fit_transform(post_df[post_cont_features])
post_df = pd.get_dummies(post_df, columns=['post_type','post_time_hour'])
### 4️⃣ Correlation Analysis
# - HeatMap
# - Variance Inflation Factor (VIF)</br>
# 📈 Range and Interpretation</br>

# | **VIF value** | **Meaning** | **Interpretation** |
# |:----------------|:-------------:|--------------------:|
# | = 1 | No correlation | The feature is independent |
# | 1–5 | Moderate correlation | Usually acceptable |
# | > 5 | High correlation | Investigate for redundancy |
# | > 10 | Very high correlation | Likely problematic |

# User-only correlation heatmap & VIF
# user_corr = user_df[user_cont_features + ['has_profile_picture']].corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(user_corr, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title("User feature correlation")
# plt.show()

# # VIF for user features
# X_user = user_df[user_cont_features + ['has_profile_picture']].values
# vif_user = pd.DataFrame({
#     'feature': user_cont_features + ['has_profile_picture'],
#     'VIF': [variance_inflation_factor(X_user, i) for i in range(X_user.shape[1])]
# })
# print(vif_user)


# # Post-only correlation heatmap & VIF
# post_corr = post_df[post_cont_features].corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(post_corr, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title("Post feature correlation")
# plt.show()

# # VIF for post features
# X_post = post_df[post_cont_features].values
# vif_post = pd.DataFrame({
#     'feature': post_cont_features,
#     'VIF': [variance_inflation_factor(X_post, i) for i in range(X_post.shape[1])]
# })
# print(vif_post)


## 5️⃣ Create Interaction DataFrame
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
df = df.merge(user_df, on='user_id', how='left')
df = df.merge(post_df, on='post_id', how='left')
df = df.reset_index(drop=True)
df.head(5)


# create user and post tensors
users = torch.tensor(df[user_cont_features + ['has_profile_picture']].to_numpy(np.float32))
posts = torch.tensor(df[post_cont_features].to_numpy(np.float32))
labels = torch.tensor(df['label'].to_numpy(np.float32))


#create the model class

class Model(nn.Module):
    
    def __init__(self, user_input_dim, post_input_dim, hidden_dim=32, drop_out_rate=0.2):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.post_tower = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, user_data, post_data):
        user_embedding = self.user_tower(user_data)
        post_embedding = self.post_tower(post_data) 
        scores = (user_embedding * post_embedding).sum(dim=1)
        pred = torch.sigmoid(scores)
        return pred

#create train test sets
train_idx, test_idx = train_test_split(range(len(labels)),test_size=0.2, random_state=42 )

user_train = users[train_idx]
user_test = users[test_idx]

post_train = posts[train_idx]
post_test = posts[test_idx]

label_train = labels[train_idx]
label_test = labels[test_idx]


#train the model
epochs = 500
hidden_dims = 64

model = Model(user_input_dim=user_train.shape[1], post_input_dim=post_train.shape[1], hidden_dim=hidden_dims, drop_out_rate=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()
losses = []
for i in range(epochs):
    model.train()
    optimizer.zero_grad()

    pred = model(user_train, post_train)
    loss = criterion(pred, label_train)
    
    losses.append(loss)
    loss.backward()
    optimizer.step()

#evaluate the model
    model.eval()
    with torch.no_grad():
        test_pred = model(user_test, post_test)
        test_loss = criterion(test_pred, label_test)
    
    if i % 10 == 0:
        print(f"Train loss: {loss}, test loss: {test_loss}")

# print(test_pred[0])
# print(label_test[0])

df_results = pd.DataFrame({
    'user_id': df.loc[test_idx, 'user_id'].values,       
    'post_id': df.loc[test_idx, 'post_id'].values,    
    'predicted': test_pred.detach().numpy(),  # convert tensor to numpy
    'actual': label_test.detach().numpy()
})

print(df_results.head())
print(df_results[df_results['predicted']!= 0.5])