# Dialogue-Contextualized Re-ranking for Medical History-Taking

[Arxiv Link](https://arxiv.org/abs/2304.01974)

If you find this code useful, please cite our work:

    @misc{zhu2023dialoguecontextualized,
      title={Dialogue-Contextualized Re-ranking for Medical History-Taking}, 
      author={Jian Zhu and Ilya Valmianski and Anitha Kannan},
      year={2023},
      eprint={2304.01974},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

### Global Reranker  

See `global_reranker/global_reranker.py`  

### Pretrained S4 Language Models  

- Autoregressive models  
See the below notebook for a demo.  
```
state-spaces-causal/load_S4.ipynb
```
- Bidirectional models

See the below notebook for a demo.  
```
state-spaces-causal/load_S4.ipynb
```

### References  
Some of the implements are modified from the following repositories.  
[allRank : Learning to Rank in PyTorch](https://github.com/allegro/allRank)  
[Structured State Spaces for Sequence Modeling](https://github.com/HazyResearch/state-spaces) 