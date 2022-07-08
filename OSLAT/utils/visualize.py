import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from pdflatex import PDFLaTeX

import pdb

def generate_tsne_plot(pairwise_distance, labels, save_path):
    transformed = TSNE(n_components=2, metric='precomputed', learning_rate='auto', random_state=0).fit_transform(pairwise_distance)
    fig = plt.figure()
    plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=labels)
    fig.savefig(save_path)


def visualize_entity_synonyms(sorted_entity_list, model, entity_inputs, save_path, n_visualize=20, n_per_entity=20):
    hidden_states, labels = [], []
    for i, (name, _) in enumerate(sorted_entity_list[:n_visualize]):
        with torch.no_grad():
            hidden_states.append( model(**{k: v[:n_per_entity].to('cuda') for k, v in entity_inputs[name].items()})[0][:, 0, :].cpu() )
        labels.extend([i] * n_per_entity)
    hidden_states = torch.cat(hidden_states, dim=0)
    pairwise_dist = pairwise_cosine_similarity(hidden_states, hidden_states).numpy()
    pairwise_dist = (pairwise_dist - pairwise_dist.min()) / (pairwise_dist.max() - pairwise_dist.min())
    generate_tsne_plot(pairwise_dist, labels, save_path)

## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.

latex_special_token = ["!@#$%^&*()"]

def generate_heatmap(text_lists, attention_lists, latex_file, headers, color='red', rescale_value = False):
    assert(len(text_lists) == len(attention_lists))
    
    colors = ['red', 'green', 'blue']
    
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
                \special{papersize=210mm,297mm}
                \usepackage{color}
                \usepackage{tcolorbox}
                \usepackage{CJK}
                \usepackage{adjustbox}
                \tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
                \begin{document}
                \begin{CJK*}{UTF8}{gbsn}'''+'\n'
        )

        for idx in range(len(text_lists)):
            text_list = text_lists[idx]
            attention_list = attention_lists[idx]

            word_num = len(text_list)
            text_list = clean_word(text_list)
            if rescale_value:
                attention_list = rescale(attention_list)

            color = colors[idx % len(colors)]

            string = f'\\textbf{{{headers[idx]}}}\n\\bigskip\n\n' 

            string += r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"

            for idx in range(word_num):
                try:
                    string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
                except:
                    break
            string += "\n}}}"
            f.write(string+'\n\n \\bigskip \n')


        f.write(r'''\end{CJK*}
                \end{document}'''
        )

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list