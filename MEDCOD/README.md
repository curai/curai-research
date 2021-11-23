# Medical Conversations

This repo contains partial code and dataset used to generate results presented in

[MEDCOD: A Medically-Accurate, Emotive, Diverse, and Controllable Dialog System](https://arxiv.org/abs/2111.09381)

which explored how we can make our automated history taking process more human-like.

## Setup

All code in this repo operates off the same virtual environment. Follow the steps below to set it up.

```
pyenv virtualenv dialogpt
pyenv local dialogpt
pip install -r requirements.txt
```


## Contents

* `0-gpt3_paraphrasing` - Scripts for paraphrasing KB questions using **GPT-3** and analysing the results
* `1-dialogpt` - All code related to data preparation and training of **DialoGPT**
* `2-dr_edits` - Code for preprocessing the Dr Edits dataset, and for training the **Empathy Prediction** model

Please see each individual folder for more details