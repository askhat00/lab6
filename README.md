# Customer Churn Prediction using Spark ML on Amazon EMR

## Overview
This project implements an end-to-end machine learning pipeline using Apache Spark ML on an Amazon EMR cluster. The goal is to predict customer churn using a real-world banking dataset and demonstrate distributed data processing, feature engineering, and model training.

## Platform
- Amazon EMR (Spark on YARN)
- Hadoop Distributed File System (HDFS)
- PySpark

## Dataset
Bank Customer Churn Dataset from Kaggle.

The dataset is uploaded to HDFS and processed in a distributed manner using Spark.

## Pipeline Stages
1. Data loading from HDFS  
2. Categorical feature encoding (Geography, Gender)  
3. Feature vector assembly  
4. Feature scaling  
5. Model training (Logistic Regression)  
6. Prediction  
7. Evaluation (Accuracy)

## Experiment
Feature ablation was performed by removing categorical features and comparing accuracy and runtime.

- With categorical features:
  - Accuracy: 0.7929
  - Runtime: ~1 minute 14 seconds

- Without categorical features:
  - Accuracy: 0.7797
  - Runtime: ~56 seconds

## How to Run
Run the Spark job from the EMR master node:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  churn_pipeline.py
