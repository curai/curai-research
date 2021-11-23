# Dr Edits

This folder contains the data and code for the **Dr Edits** section of the internship, where we looked at analysing instances where the Dr has edited the KB question with some emotional addition, and training a classifier to predict this same response.

## Contents

* `process.ipynb` - Main processing script for creating the training data

The model folder contains all code required for building the **Empathy Prediction** model

* `build_dataset.py` - Load the dataset, create train/val split, upload to W+B
* `train.py` - Construct and train the empathy prediction pipeline
* `inference.py` - Interactive inference of the empathy prediction model
* `pipeline.py` - The first version of the pipeline, which performed simple PCA over all embedding features


# Instructions

## Train/Val Split

Now that we have a nice clean dataset in the form we want, we can pass it to the `build_dataset.py` script, which will create our train/val split and upload these to W+B.

For the remainder of these code snippets, run them from inside the `model/` folder.

```
python build_dataset.py --data_path output/edits_text_filtered.csv   
```

## Train the Model

Once the dataset is uploaded to W+B, we can train our Logistic Regression-based pipeline on the dataset. To do so, simply run:

```
python train.py --dataset_artifact empathy-prediction/dataset:latest
```

This ML workflow is tightly integrated with W+B which logs and links everything together nicely. 
