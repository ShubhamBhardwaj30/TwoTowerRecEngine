
from ranker_nn import RankerNN
import torch
from torch.optim import Adam
import torch.nn as nn
import pandas as pd
import numpy as np
from data_gen import DataGenerator
from two_tower import TwoTowerModel

class Ranker:

    def __init__(self, data: DataGenerator, tower_model: TwoTowerModel):
        self.data = data
        self.test_df = data.test_df
        self.train_df = data.train_df
        self.hidden_dims = 64
        self.label_train = data.mhead_label_train
        self.label_test = data.mhead_label_test
        self.drop_out = 0.2
        self.tower_model = tower_model
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
            user_emb_all = self.tower_model.user_tower(self.data.user_train).detach().cpu().numpy()
            post_emb_all = self.tower_model.post_tower(self.data.post_train).detach().cpu().numpy()
            test_user_emb_all = self.tower_model.user_tower(self.data.user_test).detach().cpu().numpy()
            test_post_emb_all = self.tower_model.post_tower(self.data.post_test).detach().cpu().numpy()

        # Build user/post embedding maps for train
        self.user_emb_map = {uid: emb for uid, emb in zip(self.train_df["user_id"], user_emb_all)}
        self.post_emb_map = {pid: emb for pid, emb in zip(self.train_df["post_id"], post_emb_all)}
        # Build user/post embedding maps for test
        self.user_emb_map_test = {uid: emb for uid, emb in zip(self.test_df["user_id"], test_user_emb_all)}
        self.post_emb_map_test = {pid: emb for pid, emb in zip(self.test_df["post_id"], test_post_emb_all)}

        # Construct train tensor
        train_inputs = []
        for row in self.train_df.itertuples():
            if row.user_id in self.user_emb_map and row.post_id in self.post_emb_map:
                train_inputs.append(np.concatenate([self.user_emb_map[row.user_id], self.post_emb_map[row.post_id]]))
        self.train_set = torch.tensor(train_inputs, dtype=torch.float32)

        # Construct test tensor
        test_inputs = []
        for row in self.test_df.itertuples():
            if row.user_id in self.user_emb_map_test and row.post_id in self.post_emb_map_test:
                test_inputs.append(np.concatenate([self.user_emb_map_test[row.user_id], self.post_emb_map_test[row.post_id]]))
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
