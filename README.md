Machine Learning: Supervised, Unsupervised, and Reinforcement Learning
  Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every situation.
  The three major types are:
        Machine Learning
        │
        ├── 1. Supervised Learning
        │
        ├── 2. Unsupervised Learning
        │
        └── 3. Reinforcement Learning

1. Supervised Learning
  Supervised Learning is a type of machine learning where the model learns from labeled data.
  Labeled data means that the dataset contains:
    Input features (X)
    Correct output/target (y)
  The model learns the relationship:
            Input (X)  →  Output (y)

   Types of Supervised Learning
   There are two major types.
    A. Regression: Regression predicts a continuous numerical value.
       Common Regression Algorithms
          Linear Regression
          Polynomial Regression
          Decision Tree Regressor
          Random Forest Regressor
          XGBoost
          Support Vector Regression (SVR)
          Neural Networks

   B. Classification: Classification predicts a category or class.
       Common Classification Algorithms
          Logistic Regression
          Decision Tree
          Random Forest
          K-Nearest Neighbors (KNN)
          Support Vector Machine (SVM)
          Naive Bayes
          XGBoost
          Neural Networks

        Supervised Learning Workflow
                        Collect Data
                              ↓
                        Data Preprocessing
                              ↓
                        Feature Engineering
                              ↓
                        Split Data
                              ↓
                        Training Data ──→ Train Model
                              ↓
                        Testing Data ──→ Evaluate Model
                              ↓
                        Deploy Model
                              ↓
                        Predict New Data

   Evaluation Metrics for Supervised Learning
    Regression
      Common metrics:
        MAE (Mean Absolute Error)
        MSE (Mean Squared Error)
        RMSE (Root Mean Squared Error)
        R² Score
        Classification

      Common metrics:
        Accuracy
        Precision
        Recall
        F1 Score
        ROC-AUC
    For fraud detection, precision and recall are often more useful than accuracy, especially when fraud cases are rare.


2. Unsupervised Learning
    Unsupervised Learning learns patterns from unlabeled data.
    There is no correct answer column.
          Input Data (X)
                ↓
          Machine Learning Algorithm
                ↓
          Hidden Patterns
   There is no target variable.
    The algorithm tries to discover patterns automatically.

   Main Types of Unsupervised Learning
   
  A. Clustering
  Clustering groups similar data points together.
  Example:
                Customers
                    ↓
                Clustering Algorithm
                    ↓
                
         Group 1 → Young, High Spending       
         Group 2 → Older, Low Spending
         Group 3 → High Income, Medium Spending
  The algorithm creates groups based on similarities

  Common Clustering Algorithms
  1. K-Means: K-Means divides data into K clusters.
  2. DBSCAN: DBSCAN groups points based on density. It is especially useful because it can identify noise and anomalies.
  3. Hierarchical Clustering: Creates a hierarchy of clusters. Often visualized using a dendrogram.

  B. Dimensionality Reduction
      Sometimes datasets contain many features.
      Maybe there are hundreds of features.
      Dimensionality reduction reduces the number of features while preserving important information.
      Popular algorithms:
        PCA
        t-SNE
        UMAP  
      example:
                  100 Features
                      ↓
                     PCA
                      ↓
              10 Important Features
      Uses include:
        Data visualization
        Reducing complexity
        Faster ML training
        Noise reduction

  C. Association Rules
      Association learning finds relationships between items.
      Example:
          People who buy:
          Bread + Butter
          often also buy:
            Milk
      A famous example is Market Basket Analysis.
      Used in:
        E-commerce
        Retail
        Product recommendations

      Common algorithms:
        Isolation Forest
        DBSCAN
        One-Class SVM
        Autoencoders
        Gaussian Distribution
        Local Outlier Factor (LOF)

3. Reinforcement Learning:
    Reinforcement Learning (RL) is a type of machine learning where an agent learns by interacting with an environment.
    The agent takes actions and receives:
        Rewards
        Penalties
    The goal is to learn the best actions that maximize the total reward.
              Agent
                │
                │ Action
                ▼
            Environment
                │
                │ Reward + New State
                ▼
              Agent




Complete Machine Learning map:
                                  MACHINE LEARNING
                                         │
                        ┌────────────────┼────────────────┐
                        │                │                │
                        ▼                ▼                ▼
                   SUPERVISED       UNSUPERVISED     REINFORCEMENT
                        │                │                │
                        │                │                │
                   Labeled Data      Unlabeled Data    Agent + Environment
                        │                │                │
                        ▼                ▼                ▼
                   Regression        Clustering         Actions
                   Classification    Dimensionality     Rewards
                                     Reduction          Policy
                                     Anomaly Detection
                        │                │                │
                        ▼                ▼                ▼
                   Predict          Find Patterns      Learn Strategy

                   
              
