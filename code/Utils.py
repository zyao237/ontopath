from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, \
    roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score
import pandas as pd
import numpy as np

# test_results has columns: 'patient', 'drug', 'atc1', 'atc2', 'atc3', 'atc4', 'atc5', 'label', 'prediction'
def rank_eval(test_results, topk):
    test_patient_list = test_results['patient'].unique().tolist()
    eval_df = test_results.set_index('patient')
    topk_precision = {}
    topk_recall = {}
    topk_hit = {}
    topk_ndcg = {}
    for k in topk:
        topk_precision[k] = []
        topk_recall[k] = []
        topk_hit[k] = []
        topk_ndcg[k] = []
    for pat in test_patient_list:
        pat_df = eval_df.loc[pat]
        pat_df = pat_df.sort_values(['prediction'], ascending=False)
        depressant_per_pat = pat_df[pat_df['label']==1]['drug'].tolist()
        for k in topk:
            pred_list = pat_df['drug'][:k].tolist()
            topk_hit[k].append(hit(depressant_per_pat, pred_list))
            topk_ndcg[k].append(ndcg(depressant_per_pat, pred_list))
    for k in topk:
        topk_hit[k] = np.average(topk_hit[k])
        topk_ndcg[k] = np.average(topk_ndcg[k])

    return {
        'topk_hit': topk_hit,
        'topk_ndcg': topk_ndcg
    }

def hit(gt_item, pred_items):
    for i in gt_item:
        if i in pred_items:
            return 1.
    return 0.

def ndcg(gt_item, pred_items):
    topk = len(pred_items)
    ideal_rank = np.zeros(topk)
    for i in range(len(gt_item)):
        if i <len(pred_items):
            ideal_rank[i] = 1.
    pred_rank = np.zeros(topk)
    for idx, p in enumerate(pred_items):
        if p in gt_item:
            pred_rank[idx] = 1.
    return dcg(pred_rank)/dcg(ideal_rank)

def dcg(relevance_list):
    dcg = 0.
    for idx, r in enumerate(relevance_list):
        dcg += r/np.log2(idx+2.)
    return dcg

def precision_recall(gt_items, pred_items):
    score = 0
    for i in pred_items:
        if i in gt_items:
            score += 1
    return {
        'precision':score/len(pred_items),
        'recall':score/len(gt_items)
    }

# test_results has columns: 'patient', 'drug', 'atc1', 'atc2', 'atc3', 'atc4', 'atc5', 'label', 'prediction'
# testing performance on different ATC levels
def rank_eval_atc(test_results, topk, atc_level):
    test_patient_list = test_results['patient'].unique().tolist()
    eval_df = test_results.set_index('patient')
    topk_precision = {}
    topk_recall = {}
    topk_hit = {}
    topk_ndcg = {}
    for k in topk:
        topk_precision[k] = []
        topk_recall[k] = []
        topk_hit[k] = []
        topk_ndcg[k] = []
    for pat in test_patient_list:
        pat_df = eval_df.loc[pat]
        pat_df = pat_df.sort_values(['prediction'], ascending=False)
        atc_label_col = []
        true_atc = set(pat_df[pat_df['label'] == 1]['atc' + str(atc_level)].unique())
        for idx, row in pat_df.iterrows():
            if row['atc' + str(atc_level)] in true_atc:
                atc_label_col.append(1)
            else:
                atc_label_col.append(0)
        pat_df['atc_label'] = atc_label_col
        depressant_per_pat = pat_df[pat_df['atc_label']==1]['drug'].tolist()
        # assert len(depressant_per_pat) == 1, 'More than 1 drug detected as groundtruth'
        for k in topk:
            pred_list = pat_df['drug'][:k].tolist()
            topk_hit[k].append(hit(depressant_per_pat, pred_list))
            topk_ndcg[k].append(ndcg(depressant_per_pat, pred_list))
    for k in topk:
        topk_hit[k] = np.average(topk_hit[k])
        topk_ndcg[k] = np.average(topk_ndcg[k])

    return {
        'topk_hit': topk_hit,
        'topk_ndcg': topk_ndcg
    }
