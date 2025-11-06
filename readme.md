# Two Tower Recommendation Engine  
*For social media post recommendation (e.g., Facebook/Instagram) to users.*

## Overview
This codebase implements a scalable recommendation engine using a Two-Tower neural network to generate user and post embeddings, followed by a Ranker model to predict multi-head interactions (like, comment, share). The pipeline includes:

- Synthetic data generation  
- Model training  
- Evaluation  
- Embedding persistence to PostgreSQL  
- API serving for real-time recommendations  

---

## Purpose
Provide a modular, extensible framework for learning and serving embeddings to power personalized feed ranking. Easily adaptable to new features, interaction types, or downstream ranking tasks.

---

## Folder Structure
```
.
├── api/
│   └── main.py
├── src/
│   └── train/
│       ├── data_gen.py
│       ├── main.py
│       ├── ranker_nn.py
│       ├── ranker_trainer.py
│       ├── Two_Tower_Trainer.py
│       └── utils.py
├── readme.md
```
---

## Main Components

- api/main.py: Entry point for serving embeddings/models via API  
- src/train/data_gen.py: Synthetic user/post/interactions generation and preprocessing  
- src/train/main.py: End-to-end orchestration (train + evaluate + extract embeddings + persist to DB)  
- src/train/Two_Tower_Trainer.py: Two-Tower embedding model training and evaluation  
- src/train/ranker_trainer.py: Ranker training using embeddings for multi-head prediction  
- src/train/ranker_nn.py: Neural network architecture for the Ranker  
- src/train/utils.py: Utilities for data sanitization and embedding prep  

---

## Data Flow & Pipeline

1. Synthetic Data Generation: Generate user, post features, and interaction labels (liked, commented, shared)  
2. Feature Engineering: Normalize continuous features, one-hot encode categorical features  
3. Two-Tower Model Training: Learn embeddings such that dot product predicts interaction likelihood  
4. Ranker Training: Multi-head neural network using concatenated embeddings  
5. Evaluation: ROC-AUC, optimal thresholds, accuracy, Precision@K, Recall@K, NDCG@K, F1@K  
6. Embedding Extraction: Final embeddings for users and posts  
7. Database Persistence: Insert embeddings into PostgreSQL  
8. Serialization: Save model weights and scalers  


---

## ## How to Run - Docker Setup

Use the provided docker-compose.yml to simplify deployment:

    docker-compose up --build

- train: Runs the training pipeline, generates embeddings, trains models, evaluates  
- api: Serves embeddings and provides real-time recommendations  
- PostgreSQL: Stores features, interactions, and embeddings  
- Redis: Caching layer for fast retrieval  

To run only one service:

    docker-compose run train      # Only train
    docker-compose up api         # Only API

---

## Architecture

- Two-Tower Embeddings: Two independent towers encode user and post features into dense embeddings. Dot product predicts interaction probability.  
- Ranker Model: Multi-head network concatenates embeddings to predict liked, commented, shared.  
- Redis: In-memory caching of embeddings for fast retrieval.  
- PostgreSQL: Persistent storage for features, interactions, and embeddings.  
- API Serving: FastAPI exposes endpoints to query embeddings and return ranked recommendations.  

This modular architecture supports scalable training, real-time serving, and easy extensibility.

---

## Customization Options

- Adjust dataset size (num_users, num_posts) in data_gen.py  
- Add/remove features in data_gen.py  
- Change hidden dimensions, dropout, or layers in Two_Tower_Trainer.py and ranker_nn.py  
- Update training hyperparameters (epochs, lr, loss weights) in main.py or trainer classes  
- Modify evaluation metrics in evaluate_model() methods  
- Swap PostgreSQL for another database by modifying DBHelper  

---

## Example Use Cases

- Social feed ranking on social media platforms  
- Content recommendation using learned embeddings  
- Rapid A/B testing of new features  
- Downstream ML tasks like clustering or segmentation  

---

## Contact

- Author: Shubham Bhardwaj  
- GitHub: https://github.com/shubhambhardwaj30  

---

Happy experimenting!