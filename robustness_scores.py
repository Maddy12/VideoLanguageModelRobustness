import pandas as pd
import numpy as np
from tqdm import tqdm


def get_robustness_scores(df):
    """
    Requires a dataframe of the following columns:
        * R@1: Text-to-video retrieval @1
        * R@5: Text-to-video retrieval @5
        * R@10: Text-to-video retrieval @10
        * Median-R: Median retrieval rate for text-to-video retrieval 
        * Model: Model type. ex videoclip, univl_align, etc
        * Dataset: msrvtt or youcook2
        * Perturbation: Perturbation type ex. mlm_suggestion or box_jumble
        * Severity: Always 1 for text, ranges from 1-5 for video and each perturbation has a 0 for baseline clean score
        * Type: Higher level category. ex. Noise, Blur, ChangeChar, TextStyle
        * PerturbModality: video, text or video+text (multimodal)
        * Name: Perturbation but formal name, used for plotting (Optional)
        * R@1Error: Drop in performance for R@1 from clean 
        * Train: zs, ft, or scratch
        * R@5Error: Drop in performance for R@5 from clean
        * R@10Error: Drop in performance for R@10 for clean
    :param pd.DataFrame df:
    :return:
    """
    df_new = list()

    # Get GT for easier calculation
    for (idx, row) in tqdm(df.iterrows(), total=len(df), desc="Calculating Robustness"):
        cols = ['Model', 'Dataset', 'PerturbModality', 'Train']
        gt = df[df['Severity'] == 0]
        for col in cols:
            gt = gt[gt[col] == row[col]]

        for ret in ['R@1', 'R@5', 'R@10']:
            row[f'{ret}_GT'] = gt[ret].values[0]
        df_new.append(row)

    df = pd.DataFrame(df_new)
    for ret in ['R@1', 'R@5', 'R@10']:
        df[f'{ret}$\gamma^a$'] = 1 - (df[f'{ret}_GT'] - df[ret])
        df[f'{ret}$\gamma^r$'] = 1 - (df[f'{ret}_GT'] - df[ret]) / df[f'{ret}_GT']

    return df
