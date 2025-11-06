import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from two_tower import TwoTowerModel
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import os
import gc
from torch import optim
from sklearn.metrics import roc_auc_score, precision_recall_curve
import joblib
import torch
import os
from data_gen import DataGenerator

class TwoTowerTrainer:

    def __init__(self, data: DataGenerator):
        # Initialize all attributes to None
        self.data = data
        self.user_train = data.user_train
        self.post_train = data.post_train
        self.tower_label_train = data.tower_label_train
        self.mhead_label_train = data.mhead_label_train
        self.user_test = data.user_test
        self.post_test = data.post_test
        self.tower_label_test = data.tower_label_test
        self.mhead_label_test = data.mhead_label_test
        self.df = data.df
        self.train_idx = data.train_idx
        self.user_df = data.user_df
        self.post_df = data.post_df
        self.train_df = data.train_df
        self.test_df = data.test_df
        self.model = None
        self.user_embeddings = None
        self.post_embeddings = None
        self.user_embeddings_test = None
        self.post_embeddings_test = None
        self.user_dim = None
        self.post_dim = None
        self.hidden_dims = 64

    def initialize(self):
        self.user_dim = self.user_train.shape[1]
        self.post_dim = self.post_train.shape[1]

    
    def train(self, epochs=50, lr=0.05):
        
        self.initialize()
        hidden_dims = self.hidden_dims or 64
        
        self.model = TwoTowerModel(user_dim=self.user_dim,
                          post_dim=self.post_dim,
                          hidden_dim=hidden_dims,
                          dropout=0.2)

        # Device handling
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.user_train, self.post_train, self.label_train = self.user_train.to(device), self.post_train.to(device), self.tower_label_train.to(device)
        self.user_test, self.post_test, self.label_test = self.user_test.to(device), self.post_test.to(device), self.tower_label_test.to(device)

        ratio = (self.label_train==0).sum() / (self.label_train==1).sum()
        pos_weight = torch.tensor([min(ratio, 10.0)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(self.user_train, self.post_train)
            loss = criterion(logits, self.label_train)
            loss.backward()
            optimizer.step()
            # Evaluation
            self.model.eval()
            with torch.no_grad():
                test_logits = self.model(self.user_test, self.post_test)
                test_loss = criterion(test_logits, self.label_test)
                test_probs = torch.sigmoid(test_logits)  # probabilities for inspection
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}: Train loss = {loss.item():.4f}, Test loss = {test_loss.item():.4f}")
        with torch.no_grad():
            self.user_embeddings = self.model.user_tower(self.user_train)
            self.post_embeddings = self.model.post_tower(self.post_train)
            self.user_embeddings_test = self.model.user_tower(self.user_test)
            self.post_embeddings_test = self.model.post_tower(self.post_test)
        torch.cuda.empty_cache()
        gc.collect()
    

    def evaluate_model(self, top_k_list=[10, 20, 50]):
        """
        Evaluate a two-tower model for feed ranking.
        Uses raw logits for ranking to avoid threshold issues with imbalanced data.
        Computes ROC-AUC, optimal threshold, accuracy at optimal threshold, and top-k metrics.
        Top-K metrics are computed per user and averaged.
        """
        print("---------------- Two Tower Eval ----------------------")
        model = self.model
        user_test = self.user_test
        post_test = self.post_test
        label_test = self.tower_label_test
        # Get user_ids for test set
        test_idx = self.df.index.difference(self.train_idx)
        user_ids = self.df.loc[test_idx, "user_id"].values
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
        except Exception:
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

        # Store in self.metrics_df
        self.metrics_df = metrics_df
        return metrics_df

    def serialize(self, model_path="/app/models/two_tower_model.pth"):
        """
        Serialize the model and scalers to disk. Uses self.model, self.user_scaler, self.post_scaler.
        """
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
        

    def get_user_embeddings(self):
        """
        Returns user embeddings as a CPU numpy array.
        """
        if self.user_embeddings is not None:
            return self.user_embeddings.detach().cpu().numpy()
        return None

    def get_post_embeddings(self):
        """
        Returns post embeddings as a CPU numpy array.
        """
        if self.post_embeddings is not None:
            return self.post_embeddings.detach().cpu().numpy()
        return None
    
    def get_training_data(self):
        return self.df, self.train_idx 