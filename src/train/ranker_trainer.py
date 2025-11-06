
from ranker_nn import RankerNN
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from data_gen import DataGenerator
from two_tower_trainer import TwoTowerTrainer
import os
# Additional imports for evaluation
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve

class Ranker:

    def __init__(self, data: DataGenerator, tower: TwoTowerTrainer):
        self.data = data
        self.test_df = data.test_df
        self.train_df = data.train_df
        self.hidden_dims = 64
        self.label_train = data.mhead_label_train
        self.label_test = data.mhead_label_test
        self.drop_out = 0.2
        self.tower = tower
        self.tower_model = tower.model
        self.train_set = None
        self.test_set = None
        self.input_dims = None
        self.output_dims = None
        self.model = None
        self.user_emb_map = {}
        self.post_emb_map = {}

    def initialize(self):
        # Switch model to eval mode
        self.tower_model.eval()
        with torch.no_grad():
            user_emb_all = self.tower.user_embeddings.detach().cpu().numpy()
            post_emb_all = self.tower.post_embeddings.detach().cpu().numpy()
            test_user_emb_all = self.tower.user_embeddings_test.detach().cpu().numpy()
            test_post_emb_all = self.tower.post_embeddings_test.detach().cpu().numpy()

        # Build user/post embedding maps for train
        self.user_emb_map = {uid: emb for uid, emb in zip(self.train_df["user_id"], user_emb_all)}
        self.post_emb_map = {pid: emb for pid, emb in zip(self.train_df["post_id"], post_emb_all)}
        # Build user/post embedding maps for test
        self.user_emb_map_test = {uid: emb for uid, emb in zip(self.test_df["user_id"], test_user_emb_all)}
        self.post_emb_map_test = {pid: emb for pid, emb in zip(self.test_df["post_id"], test_post_emb_all)}

        # Construct train tensor
        user_emb_array = np.vstack([self.user_emb_map[uid] for uid in self.train_df["user_id"]])
        post_emb_array = np.vstack([self.post_emb_map[pid] for pid in self.train_df["post_id"]])
        train_inputs = np.hstack([user_emb_array, post_emb_array])
        self.train_set = torch.tensor(train_inputs, dtype=torch.float32)

        # Construct test tensor
        user_emb_array = np.vstack([self.user_emb_map_test[uid] for uid in self.test_df["user_id"]])
        post_emb_array = np.vstack([self.post_emb_map_test[pid] for pid in self.test_df["post_id"]])
        test_inputs = np.hstack([user_emb_array, post_emb_array])
        self.test_set = torch.tensor(test_inputs, dtype=torch.float32)

        self.input_dims = self.train_set.shape[1]
        self.output_dims = self.label_train.shape[1]
        self.model = RankerNN(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            dropout=self.drop_out
        )



    def train(self, epoch=50, lr=0.001):
        self.initialize()
        optimizer = Adam(self.model.parameters(), lr=lr)
        pos_weights = (self.label_train == 0).sum(axis=0) / (self.label_train == 1).sum(axis=0)
        pos_weights = torch.tensor(np.minimum(10, pos_weights), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        for e in range(epoch):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(self.train_set)
            loss = criterion(logits, self.label_train)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                test_logits = self.model(self.test_set)
                test_loss = criterion(test_logits, self.label_test)

            if e % 10 == 0:
                print(f"Epoch {e+1}: Train loss = {loss.item():.4f}, Test loss = {test_loss.item():.4f}")
        

    def serialize(self, model_path="/app/models/ranker.pth"):
        """
        Serialize the model and scalers to disk. Uses self.model, self.user_scaler, self.post_scaler.
        """
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")


    def evaluate_model(self, top_k_list=[10, 20, 50]):
        """
        Evaluate the multi-head ranker model for feed ranking.
        Computes ROC-AUC, optimal threshold, accuracy, and top-k metrics for each head and overall.
        """
        print("---------------- Ranker Eval ----------------------")
        model = self.model
        test_set = self.test_set
        label_test = self.label_test
        # Get user_ids for test set
        test_idx = self.data.df.index.difference(self.data.train_idx)
        user_ids = self.data.df.loc[test_idx, "user_id"].values
        # Head names
        head_names = ['liked', 'commented', 'shared']
        model.eval()
        with torch.no_grad():
            logits = model(test_set)
            probs = torch.sigmoid(logits).cpu().numpy()
            logits_np = logits.cpu().numpy()
            labels = label_test.cpu().numpy()
            user_ids_np = np.array(user_ids)

        table_rows = []
        aucs = []
        # Per-head metrics
        for i, head in enumerate(head_names):
            try:
                auc = roc_auc_score(labels[:, i], probs[:, i])
            except Exception:
                auc = np.nan
            aucs.append(auc)
            precision, recall, thresholds = precision_recall_curve(labels[:, i], probs[:, i])
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5
            accuracy_opt = ((probs[:, i] >= optimal_threshold).astype(int) == labels[:, i]).mean()
            table_rows.append({'metric': f'roc_auc_{head}', 'value': auc})
            table_rows.append({'metric': f'optimal_threshold_{head}', 'value': optimal_threshold})
            table_rows.append({'metric': f'accuracy_at_optimal_threshold_{head}', 'value': accuracy_opt})

        # Average ROC-AUC
        avg_auc = np.nanmean(aucs)
        table_rows.append({'metric': 'roc_auc_mean', 'value': avg_auc})

        # Top-K metrics using mean probability across heads as ranking score
        mean_probs = probs.mean(axis=1)
        # For overall label: at least one head positive
        overall_labels = (labels.sum(axis=1) > 0).astype(int)
        unique_users = np.unique(user_ids_np)
        for k in top_k_list:
            precisions, recalls, ndcgs, f1s = [], [], [], []
            for uid in unique_users:
                idx = np.where(user_ids_np == uid)[0]
                if len(idx) == 0:
                    continue
                user_scores = mean_probs[idx].flatten()
                user_labels = overall_labels[idx].flatten()
                if len(user_labels) == 0:
                    continue
                # Top-K indices for this user
                top_k_user_idx = np.argsort(-user_scores)[:min(k, len(user_scores))]
                top_k_labels = user_labels[top_k_user_idx]
                precision_at_k = top_k_labels.sum() / min(k, len(user_scores))
                recall_at_k = top_k_labels.sum() / user_labels.sum() if user_labels.sum() > 0 else 0
                discounts = 1 / np.log2(np.arange(2, min(k, len(user_scores)) + 2))
                dcg = (top_k_labels * discounts).sum()
                idcg = (np.sort(user_labels)[-min(k, len(user_scores)):][::-1] * discounts).sum()
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

        metrics_df = pd.DataFrame(table_rows)
        print("\nRanker Evaluation Metrics Summary:")
        print(metrics_df.to_string(index=False))
        self.metrics_df = metrics_df
        return metrics_df