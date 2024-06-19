import json
from tqdm import tqdm
import os
import fire
import re
import string
import unicodedata
import numpy as np

from collections import Counter


def normalize_text(s):
    s = unicodedata.normalize('NFD', s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        # Unlike most tasks, QReCC and SQuAD-2.0 assign 1.0 in this edge case. We don't for uniformity.
        print("\n#> F1 Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\n")
        return 1

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def EM(answer, references):
    if answer == '':
        return False
    normalized_answer = normalize_text(answer)
    normalized_references = [normalize_text(ref) for ref in references]
    for reference in normalized_references:
        if reference == normalized_answer:
            return True
    return False


def F1(prediction, answers_list):
    assert type(answers_list) == list

    return max(f1_score(prediction, ans) for ans in answers_list)


def Eval(file_path=''):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cnt = 0
    f1_list = []
    for d in data:
        LLM_answer = d["LLM_answer"]
        reference = d["reference"]
        if EM(LLM_answer, reference):
            d["Eval"] = "Correct"
            cnt += 1
        else:
            d["Eval"] = "Wrong"
        d['f1_score'] = F1(LLM_answer, reference)
        f1_list.append(d['f1_score'])

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open(file_path.replace('.json', '_accuracy').replace('/details', ''), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {cnt/len(data):.4f}\n")
        f.write(f"EM: {cnt}/{len(data)}\n")
        f.write(f"F1: {np.mean(f1_list):.4f}\n")


if __name__ == "__main__":
    fire.Fire(Eval)
