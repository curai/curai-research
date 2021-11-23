import argparse
import os
import pickle
import random
import warnings

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

from tqdm import tqdm
import wandb

from utils import get_datetime_string
from pipeline import EmbedTransform, MessagePCA

warnings.filterwarnings('ignore')
tqdm.pandas()
random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_artifact", default="dialogpt/dialogpt:latest", type=str)
    return parser.parse_args()

def download_data(dataset_artifact):
    experiment_name = f"Train Model: {get_datetime_string()}"
    train_filename, val_filename = "dataset_train.pkl", "dataset_val.pkl"
    run = wandb.init(
        project="empathy-prediction",
        sync_tensorboard=True,
        config=vars(args), # Automatically log all config vars
        name=experiment_name,
        save_code=False,
        job_type="train")
    artifact = run.use_artifact(dataset_artifact, type='dataset')
    artifact_dir = artifact.download()
    dataset_path = f"{artifact_dir}/"

    with open(dataset_path + train_filename, 'rb') as handle:
        df_trn = pickle.load(handle)
    with open(dataset_path + val_filename, 'rb') as handle:
        df_val = pickle.load(handle)

    return df_trn, df_val

def save_model(model, filename="model.pkl"):
    with open(filename, mode='wb') as f:
        pickle.dump(model, f)

    model_artifact = wandb.Artifact('model', type='model')
    model_artifact.add_file(filename)
    wandb.run.log_artifact(model_artifact)

if __name__ == "__main__":
    args = parse_args()

    # Load train/test data from WandB

    df_train, df_val = download_data(args.dataset_artifact)

    label_col_name = "empathy_label"

    train_x = df_train.drop(label_col_name, axis=1)
    train_y = df_train[label_col_name]

    val_x = df_val.drop(label_col_name, axis=1)
    val_y = df_val[label_col_name]

    parameters = {
            'pca__n_components': [10, 15, 20, 25, 30, 35, 40, 50, 60, 70],
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__class_weight': [
                {0:1, 1:1, 2:1, 3:1},
                {0:1, 1:1, 2:10, 3:10},
                {0:1, 1:2, 2:10, 3:10},
                {0:1, 1:3, 2:10, 3:10},
                {0:1, 1:10, 2:10, 3:10}
            ]
        }
    child_pipeline = Pipeline([('pca', MessagePCA()), ('clf', LogisticRegression(max_iter=1000))])
    cv_pipeline = GridSearchCV(child_pipeline, parameters, n_jobs=-1, verbose=1, scoring="f1_macro")
    pipeline = Pipeline([('embed', EmbedTransform()), ('child', cv_pipeline)])

    pipeline.fit(train_x, train_y)

    save_model(pipeline)

    # Evaluate the model
    labels = ["None", "Affirmative", "Empathy", "Apology"]
    preds = pipeline.predict(val_x)
    preds_prob = pipeline.predict_proba(val_x)

    report = classification_report(val_y, 
        preds, 
        labels=[0, 1, 2, 3], 
        target_names=labels,
        output_dict=True)
    # print(report)
    wandb.log({'perf_metrics': report}, commit=False)

    df = pd.DataFrame(report).transpose()
    wandb.log({'classification_report': wandb.Html(df.to_html(float_format="%.2f"))}, commit=False)

    wandb.sklearn.plot_classifier(pipeline, train_x, val_x, train_y, val_y, preds, preds_prob, 
                                labels, model_name='LogisticRegression', feature_names=None)
