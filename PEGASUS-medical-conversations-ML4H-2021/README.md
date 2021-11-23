# [Adding more data does not always help: A study in medical conversation summarization with PEGASUS](https://arxiv.org/abs/2111.07564)

Authors: Varun Nair, Namit Katariya, Xavier Amatriain, Ilya Valmianski, Anitha Kannan

Abstract: Medical conversation summarization is an important component of capturing information from encounters between patients and physicians that can be used to provide care in the future and facilitate patient hand-offs between providers. Summaries, however, can be time-consuming to produce and require domain expertise. Modern pre-trained NLP models such as PEGASUS have emerged as capable alternatives to human summarization, reaching state-of-the-art performance on many summarization benchmarks. However, many downstream tasks still require at least moderately sized datasets to achieve satisfactory performance. In this work we (1) explore the effect of dataset size on transfer learning medical conversation summarization using PEGASUS and (2) evaluate various iterative labeling strategies in the low-data regime, following their success in the classification setting. We find that model performance saturates with increase in dataset size and that the various active-learning strategies evaluated all show equivalent performance consistent with simple dataset size increase. We also find that naive iterative pseudo-labeling is on-par or slightly worse than no pseudo-labeling. Our work sheds light on the successes and challenges of translating low-data regime techniques in classification to medical conversation summarization and helps guides future work in this space.

### Experiments

The experiments in our ML4H submission were generated using the script ```$ bash run.sh``` (*warning data not included*).

This script contains the necessary code to run all experiments shown in Figure 1 and will log all metrics appropriately to Weights and Biases. To dive into the specific file containing the code for PEGASUS, see ```train_pegasus.py``` and ```summary/pegasus/trainer.py```.

### TODO

These scripts still contain several dependencies to Curai Health specific libraries, which will be removed and refactored in the future. Note we also do not release the dataset used for training as part of patient privacy protections.