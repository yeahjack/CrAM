import os
import fire

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# import torch
from tqdm import tqdm
import json
import numpy as np

from utils.RAG import RAG
from utils.find_best_heads import find_top_k_heads
from eval import Eval


def Log_gen(sample, prompt, result):
    return {
        "id": sample['id'],
        "prompt": prompt,
        "question": sample["question"],
        "LLM_answer": result,
        "reference": sample["reference"],
        "wrong answer": sample["wrong answer"],
    }


def RAG_gen(
        filepath: str = "nq_1000_bge.json",
        # LLM: str = "llama2_13b_chat",
        LLM: str = "llama3",
        # LLM: str = "cag",
        type: str = "with_contexts",
        # type: str = "with_contexts_cag",
        # type: str = "with_contexts_score",
        contexts_type: list = ["ori_fake", "reranked_dense"],
        # contexts_type: list = ["reranked_dense"],
        whether_my_decoding: bool = False,
        whether_re_weighting: bool = False,
        whether_delete_false_paras: bool = False,
        whether_ideal_scores: bool = False,
        ideal_score: int = 0,
        modify_all_layers: bool = False,
        heads_topk: int = 559,
        whether_relevant: bool = False,
        topk: int = 4,
        fake_num: int = 1,
        causal_num: int = 100):

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if whether_re_weighting and not modify_all_layers:
        if causal_num == 100:
            input_path = os.path.join(f'dataset', filepath.split('_')[0], LLM)
        else:
            input_path = os.path.join(f'Head_data{causal_num}', filepath.split('_')[0], LLM)
        input_filepath = os.path.join(input_path, 'selected_heads.json')
        find_top_k_heads(input_path=input_path, topk=heads_topk)
        with open(input_filepath, "r", encoding="utf-8") as f:
            layers_to_be_modified = json.load(f)
    else:
        layers_to_be_modified = dict()
    RAG_model = RAG(LLM, normalize_L2=True, whether_my_decoding=whether_my_decoding, whether_re_weighting=whether_re_weighting, layers_to_be_modified=layers_to_be_modified)

    prompt_and_answer = []
    for sample in tqdm(data):
        contexts = []
        if whether_relevant == True:
            relevant_scores = []
        truthful_scores = []
        scores = []
        if "ori_fake" in contexts_type:
            contexts.extend(sample["ori_fake"][:fake_num])
            if whether_re_weighting == True or whether_delete_false_paras == True or type == "with_contexts_score" or type == "with_contexts_cag":
                if whether_ideal_scores:
                    truthful_scores.extend([ideal_score] * fake_num)
                    if whether_relevant == True:
                        relevant_scores.extend([10] * fake_num)
                else:
                    truthful_scores.extend(sample["ori_fake_truthful_scores"][:fake_num])
                    if whether_relevant == True:
                        relevant_scores.extend(sample["ori_fake_relevant_scores"][:fake_num])
        if "reranked_dense" in contexts_type:
            contexts.extend(sample["reranked_dense_ctxs"][:topk])
            if whether_re_weighting == True or whether_delete_false_paras == True or type == "with_contexts_score" or type == "with_contexts_cag":
                if whether_ideal_scores:
                    truthful_scores.extend([10] * topk)
                    if whether_relevant == True:
                        relevant_scores.extend([10] * topk)
                else:
                    truthful_scores.extend(sample["reranked_dense_ctxs_truthful_scores"][:topk])
                    if whether_relevant == True:
                        relevant_scores.extend(sample["reranked_dense_ctxs_relevant_scores"][:topk])
        if whether_re_weighting == True and whether_delete_false_paras == False:
            if whether_relevant == True:
                scores = np.array([int(score) for score in truthful_scores]) * np.array([int(score) for score in relevant_scores])
            else:
                scores = np.array([int(score) for score in truthful_scores])
                if whether_ideal_scores == False:
                    scores -= np.min(scores)
            if np.max(scores) != 0:
                scores = (scores / np.max(scores))
            else:
                scores = (scores + 1)
            combined = list(zip(scores, contexts))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            scores, contexts = zip(*combined_sorted)
            scores = list(scores)
            contexts = list(contexts)
        if type == "with_contexts_score" or type == "with_contexts_cag":
            if whether_relevant == True:
                scores = np.array([int(score) for score in truthful_scores]) * np.array([int(score) for score in relevant_scores])
            else:
                scores = np.array([int(score) for score in truthful_scores])
            combined = list(zip(scores, contexts))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            scores, contexts = zip(*combined_sorted)
            scores = list(scores)
            contexts = list(contexts)
        if whether_delete_false_paras == True:
            if whether_relevant == True:
                scores = np.array([int(score) for score in truthful_scores]) * np.array([int(score) for score in relevant_scores])
            else:
                scores = np.array([int(score) for score in truthful_scores])
            contexts = list(filter(lambda x: scores[contexts.index(x)] > 3, contexts))

        if type == "with_contexts":
            prompt, result = RAG_model.run_RAG(question=sample["question"], paras=contexts, scores=scores)
            prompt_and_answer.append(Log_gen(sample=sample, prompt=prompt, result=result))
        elif type == "without_contexts":
            prompt, result = RAG_model.run_RAG_without_contexts(question=sample["question"])
            prompt_and_answer.append(Log_gen(sample=sample, prompt=prompt, result=result))
        elif type == "with_contexts_score" or type == "with_contexts_cag":
            prompt, result = RAG_model.run_RAG(question=sample["question"], paras=contexts, scores=scores, type=type)
            prompt_and_answer.append(Log_gen(sample=sample, prompt=prompt, result=result))

        print(prompt + '\n' + result)
        if whether_re_weighting or type == "with_contexts_score" or type == "with_contexts_cag" or whether_delete_false_paras == True:
            if whether_ideal_scores == False:
                folder = f"./results_real/{filepath.split('.')[0]}/{LLM}/{type}/details"
            elif causal_num == 100:
                folder = f"./results{ideal_score}/{filepath.split('.')[0]}/{LLM}/{type}/details"
            else:
                folder = f"./results{ideal_score}_{causal_num}/{filepath.split('.')[0]}/{LLM}/{type}/details"
        else:
            folder = f"./results/{filepath.split('.')[0]}/{LLM}/{type}/details"
        # filename = f"{('_'.join(contexts_type) +f'_top{topk}' if type!='without_contexts' else 'no_contexts')+('_correction' if corpus_correction==True else '')+('_decoding' if whether_my_decoding==True else '')+('_weighting' if whether_re_weighting==True else '')+('_delete' if whether_delete_false_paras==True else '')}_{a}_{b}.json"
        names_parts = []
        if type == 'without_contexts':
            names_parts.append('no_contexts')
        elif type == "with_contexts_score" or type == "with_contexts_cag":
            names_parts.append(type)
            if whether_ideal_scores:
                names_parts.append('ideal')
            if fake_num != 1:
                names_parts.append('ori_fake')
        else:
            names_parts.append('_'.join(contexts_type) + f'_top{topk}')
            if whether_my_decoding:
                names_parts.append('decoding')
            if whether_delete_false_paras:
                names_parts.append('delete')
            if whether_re_weighting:
                names_parts.append('weighting')
                if whether_ideal_scores:
                    names_parts.append('ideal')
                if not modify_all_layers and whether_re_weighting:
                    names_parts.append(f'heads{heads_topk}')
        filename = '_'.join(names_parts) + '.json'
        if fake_num != 1:
            filename = filename.replace('ori_fake', f"ori_fake_{fake_num}")
        file_path = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file_path, 'w', encoding="utf8") as file:
            json.dump(prompt_and_answer, file, ensure_ascii=False, indent=4)

    Eval(file_path)


if __name__ == "__main__":
    fire.Fire(RAG_gen)
