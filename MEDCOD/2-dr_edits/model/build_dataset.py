import wandb
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_datetime_string
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.data_path, na_filter=False)
    df.columns = ["prev_dr_turn", "prev_pt_resp", "curr_dr_turn", "empathy_label"]
    df_train, df_val = train_test_split(df, random_state=42, test_size=0.2)

    print(f"Split {df.shape[0]} instances into train:{df_train.shape[0]}, val:{df_val.shape[0]}")

    # TODO log more summary statistics (class distribution, num instances, shape, etc.)

    # Save the datasets to files
    with open('dataset.pkl', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('dataset_train.pkl', 'wb') as handle:
        pickle.dump(df_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('dataset_val.pkl', 'wb') as handle:
        pickle.dump(df_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Upload the files to WandB
    run = wandb.init(name=f"Build Dataset: {get_datetime_string()}", 
        project="empathy-prediction", 
        config=vars(args),
        job_type="dataset-creation") # IMPORTANT: allows us to group in the W+B interface

    artifact = wandb.Artifact('dataset', type='dataset')
    
    artifact.add_file("dataset.pkl")
    artifact.add_file("dataset_train.pkl")
    artifact.add_file("dataset_val.pkl")

    # Allow us to inspect the training dataset from the W+B interface
    dataset_table = wandb.Table(dataframe=df_train)
    artifact.add(dataset_table, "train")

    run.log_artifact(artifact)