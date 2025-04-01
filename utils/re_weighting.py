import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
from functools import partial
from tqdm import tqdm
import numpy as np
from prompt import get_prompt

RAG_prompt1 = """Given the following information: \n"""
RAG_prompt2 = """Answer the following question based on the given information or your internal knowledge with one or few words without the source.
Question: {question}
Answer: {answer}"""
# RAG_prompt2 = """Answer the following question based on the given information or your internal knowledge.
# Question: {question}
# Answer: {answer}"""


class Re_Weighting_Strategy:

    def __init__(self, model_name: str = "Llama-2-13b-chat-hf", layers_to_be_modified: dict = dict(), bad_words_ids=[]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.model_name = model_name
        self.bad_words_ids = bad_words_ids
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.model_num_attention_heads = self.model.config.num_attention_heads
        if not layers_to_be_modified:
            layers_to_be_modified = {i: list(range(self.model_num_attention_heads)) for i in range(self.num_hidden_layers)}
        self.layers_to_be_modified = layers_to_be_modified

    def edit_attention_mask(self, module: torch.nn.Module, input_args: tuple, input_kwargs: dict, attention_weight: list, head_idx: list = []):
        weight_len = attention_weight.size()[-1]
        dtype, device = input_kwargs['hidden_states'].dtype, input_kwargs['hidden_states'].device
        if input_kwargs['attention_mask'] == None:
            bsz, head_dim = 1, 1
            tgt_len = input_kwargs['hidden_states'].size()[1]
            src_len = input_kwargs['position_ids'][0][-1] + 1
            if tgt_len == 1:
                attention_mask = torch.zeros([bsz, head_dim, tgt_len, src_len], dtype=dtype, device=device)
            else:
                min_value = torch.finfo(dtype).min
                upper_triangle_matrix = torch.triu(torch.full((tgt_len, src_len), min_value, dtype=dtype, device=device), diagonal=1)
                attention_mask = upper_triangle_matrix.unsqueeze(0).unsqueeze(0).expand(bsz, head_dim, tgt_len, src_len)
        else:
            attention_mask = input_kwargs['attention_mask'].clone()
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        if head_dim == 1:
            attention_mask = attention_mask.repeat(1, self.model_num_attention_heads, 1, 1)
            head_dim = self.model_num_attention_heads
        # dtype, device = attention_mask.dtype, attention_mask.device
        expanded_weight = attention_weight.unsqueeze(0).unsqueeze(0).repeat(bsz, head_dim, tgt_len, 1).to(dtype=dtype, device=device)
        mask = (attention_mask[..., :weight_len] == 0.0)
        for h in head_idx:
            attention_mask[:, h, :, :weight_len][mask[:, h, :, :]] = expanded_weight[:, h, :, :][mask[:, h, :, :]]
        input_kwargs['attention_mask'] = attention_mask
        return input_args, input_kwargs

    def decode_with_special_attention(self, question: str = '', paras: list = [], scores: list = [], answer: str = ''):
        add_special_tokens = True
        if self.model_name.find("Llama-3") != -1:
            prompt = get_prompt(context=paras, question=question, answer='', type='with_contexts')
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant!"
                },
                {
                    "role": "user",
                    "content": f"{prompt}"
                },
            ]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            if answer != '':
                prompt += answer
        elif self.model_name.find('Qwen') != -1:
            prompt = get_prompt(context=paras, question=question, answer='', type='with_contexts')
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"{prompt}"
                },
            ]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            add_special_tokens = False
            if answer != '':
                prompt += answer
        else:
            prompt = get_prompt(context=paras, question=question, answer=answer, type='with_contexts')
        model_inputs = self.tokenizer([prompt], return_tensors="pt", return_offsets_mapping=True, add_special_tokens=add_special_tokens).to("cuda")
        attention_weight = model_inputs['attention_mask'].clone().to(torch.float16)
        for i, p in enumerate(paras):
            para = ("Passage-%d: " % i) + p + '\n'
            start_idx = prompt.find(para)
            end_idx = start_idx + len(para) - 1
            start_id_pos = None
            end_id_pos = None
            for idx, x in enumerate(model_inputs['offset_mapping'][0]):
                if start_idx >= x[0]:
                    start_id_pos = idx
                if end_idx >= x[0]:
                    end_id_pos = idx
            attention_weight[:, start_id_pos:end_id_pos + 1] = torch.full((1, end_id_pos + 1 - start_id_pos), scores[i]).to("cuda").to(torch.float16)
        model_inputs.pop('offset_mapping')
        return model_inputs, attention_weight

    @torch.no_grad()
    def run_RAG_with_attention_weighting(self, question: str = '', paras: list = [], scores: list = []):
        model_inputs, attention_weight = self.decode_with_special_attention(question=question, paras=paras, scores=scores)
        registered_hooks = []

        for layer_idx, head_idx in self.layers_to_be_modified.items():
            module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
            hook_func = partial(self.edit_attention_mask, attention_weight=torch.log(attention_weight), head_idx=head_idx)
            registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
            registered_hooks.append(registered_hook)

        prompt = self.tokenizer.decode(model_inputs['input_ids'][0][1:])
        para_dict = {"do_sample": False, "max_new_tokens": 100}
        if self.bad_words_ids:
            para_dict["bad_words_ids"] = self.bad_words_ids
        if self.model_name.find("Llama-3") != -1:
            para_dict["eos_token_id"] = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = self.model.generate(**model_inputs, **para_dict)
        output = self.tokenizer.decode(outputs[0][1:-1])
        prompt_end_index = output.find(prompt) + len(prompt)
        output = output[prompt_end_index:]
        for registered_hook in registered_hooks:
            registered_hook.remove()
        return prompt, output


class Find_Best_Heads(Re_Weighting_Strategy):

    def __init__(self, model_name: str = "Llama-2-13b-chat-hf", layers_to_be_modified: list = []):
        super().__init__(model_name=model_name)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model_num_hidden_layers = self.model.config.num_hidden_layers
        # self.model_num_hidden_layers = 2
        self.model_num_attention_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def cal_logits(self, question: str = '', paras: list = [], scores: list = [], right_answer: str = '', wrong_answer: str = 'my name is'):
        answer = wrong_answer
        answer_ids = self.tokenizer([answer], return_tensors="pt")['input_ids'][0, 1:]
        model_inputs, attention_weight = self.decode_with_special_attention(question=question, paras=paras, scores=scores, answer=answer)
        registered_hooks = []

        self.ori_logits = self.model(**model_inputs, return_dict=True)['logits'].clone()
        self.ori_prob_sum = self.ori_logits[0, -len(answer_ids):][np.arange(len(answer_ids)), answer_ids].sum()
        prob_change = []
        for layer_idx in tqdm(range(self.model_num_hidden_layers)):
            module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
            prob_change_layer = []
            for head_idx in range(self.model_num_attention_heads):
                hook_func = partial(self.edit_attention_mask, attention_weight=torch.log(attention_weight), head_idx=[head_idx])
                registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
                registered_hooks.append(registered_hook)
                current_logits = self.model(**model_inputs, return_dict=True)['logits'].clone()
                current_prob_sum = current_logits[0, -len(answer_ids):][np.arange(len(answer_ids)), answer_ids].sum()
                prob_change_layer.append((self.ori_prob_sum - current_prob_sum).item())
                for registered_hook in registered_hooks:
                    registered_hook.remove()
            prob_change.append(prob_change_layer)
        return prob_change

if __name__ == "__main__":
    # 导入必要的模块
    import time
    
    # 初始化Re_Weighting_Strategy实例
    # 注意：实际运行时，应该选择一个适合你计算资源的模型
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # 或者使用更小的模型以便快速演示
    
    print("正在加载模型，这可能需要几分钟时间...")
    rag_strategy = Re_Weighting_Strategy(model_name=model_name)
    
    # 准备示例问题和相关段落
    question = "谁是苹果公司的创始人？"
    paras = [
        "苹果公司（Apple Inc.）是由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩在1976年创立的美国跨国技术公司。",
        "微软公司是由比尔·盖茨和保罗·艾伦于1975年创立的美国跨国技术公司。",
        "史蒂夫·乔布斯（Steve Jobs，1955年2月24日－2011年10月5日）是苹果公司的联合创始人和前CEO。"
    ]
    # 段落的相关性分数（越高越相关）
    scores = [0.8, 0.3, 0.9]
    
    print("\n===== 基本RAG演示 =====")
    print(f"问题: {question}")
    print("相关段落:")
    for i, (para, score) in enumerate(zip(paras, scores)):
        print(f"段落{i+1} (相关性: {score:.2f}): {para[:50]}...")
    
    # 运行基本的RAG过程
    print("\n正在使用默认注意力头配置生成答案...")
    start_time = time.time()
    prompt, output = rag_strategy.run_RAG_with_attention_weighting(
        question=question, paras=paras, scores=scores
    )
    end_time = time.time()
    print(f"生成时间: {end_time - start_time:.2f}秒")
    print(f"答案: {output}")
    
    print("\n===== 寻找最佳注意力头 =====")
    # 使用Find_Best_Heads找出最有效的注意力头
    print("正在评估不同注意力头的效果（这可能需要较长时间）...")
    best_heads_finder = Find_Best_Heads(model_name=model_name)
    
    # 设置正确答案和错误答案用于评估
    right_answer = "史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩"
    wrong_answer = "比尔·盖茨"
    
    start_time = time.time()
    prob_changes = best_heads_finder.cal_logits(
        question=question, paras=paras, scores=scores, 
        right_answer=right_answer, wrong_answer=wrong_answer
    )
    end_time = time.time()
    print(f"评估时间: {end_time - start_time:.2f}秒")
    
    # 找出影响最大的前5个头
    flat_changes = [(layer, head, change) for layer, layer_changes in enumerate(prob_changes) 
                   for head, change in enumerate(layer_changes)]
    top_5_heads = sorted(flat_changes, key=lambda x: x[2], reverse=True)[:5]
    
    print("\n影响最大的5个注意力头:")
    for layer, head, change in top_5_heads:
        print(f"层: {layer}, 头: {head}, 变化量: {change:.4f}")
    
    # 使用最有效的头配置再次运行RAG
    best_layers_to_be_modified = {layer: [head] for layer, head, _ in top_5_heads}
    print("\n正在使用优化后的注意力头配置生成答案...")
    
    optimized_rag_strategy = Re_Weighting_Strategy(
        model_name=model_name, 
        layers_to_be_modified=best_layers_to_be_modified
    )
    
    start_time = time.time()
    prompt, optimized_output = optimized_rag_strategy.run_RAG_with_attention_weighting(
        question=question, paras=paras, scores=scores
    )
    end_time = time.time()
    print(f"生成时间: {end_time - start_time:.2f}秒")
    print(f"优化后答案: {optimized_output}")
    
    print("\n===== 结果比较 =====")
    print(f"默认配置答案: {output}")
    print(f"优化后答案: {optimized_output}")