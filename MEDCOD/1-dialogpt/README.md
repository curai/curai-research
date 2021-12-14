# Medical Cconversation Simulator with DialoGPT

This folder contains the code for preprocessing data, training DialoGPT, and evaluating it in an interactive setting.

Example of datasets used can be seen in dataset_* files.

## Train: Finetuning DialoGPT using our medical dataset
The following hyperparameters work on a single K80. These hyperparameters are also specified in `train.sh`, which is the recommended way to invoke training.

## Inference: Play with the chatbot

```bash
python chatbot.py --model_artifact medical-conversations/dialogpt-model:paraphrased_emoted_short # or do --model_artifact dialogpt/dialogpt-model:latest
```

The model expects the previous customer response, an Emote control code, and a Finding control code, e.g.
```
                >>> Customer: Yes. Emote: empathy. Finding: productive cough
Next Response: I am sorry about that. Is your cough productive, bringing up phlegm or mucus?
                >>> Customer: Yes. Emote: empathy. Finding: productive cough
Next Response: Sorry to know that. Do you have a cough that brings up phlegm or mucus?

                >>> Customer: Yes. Emote: none. Finding: eyelid erythema
Next Response: Do you have any eyelid redness?
                >>> Customer: Yes. Emote: none. Finding: eyelid erythema
Next Response: Is there any redness around your eyelids?
```
