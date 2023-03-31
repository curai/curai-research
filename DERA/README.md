# DERA: Enhancing Large Language Model Completions with  Dialog-Enabled Resolving Agents

[Arxiv Link](https://arxiv.org/abs/2303.17071)

If you find this dataset useful, please cite the following paper;

    @misc{nair2023dera,
          title={DERA: Enhancing Large Language Model Completions with Dialog-Enabled Resolving Agents}, 
          author={Varun Nair and Elliot Schumacher and Geoffrey Tso and Anitha Kannan},
          year={2023},
          eprint={2303.17071},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }

The following repository contains the open-ended question-answering version of [MedQA](https://github.com/jind11/MedQA).  These consists of questions that were rewritten using a GPT-4 prompt, using the approach described in the paper.  These were not manually rewritten by human annotators, so there may be some inconsistencies.

Notes
- In each file, the open-ended question is included for each question-answer pair in the "question_open" field.
- The file format is the same as the original except for that additional field.  Also, note that the training file was converted from the 4-option format, while the others are from the 5-option format.  This does not matter for open-ended evaluation, but be aware that they have different amounts of unused options.
- Please see LICENSE.txt and LICENSE_MEDQA.txt (original MedQA license).

