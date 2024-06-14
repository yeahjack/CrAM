import re
from importlib.metadata import version
from packaging import version as pkg_version
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers

from .prompt import *
from .openai_api import askChatGPT
from colorama import Fore, Style
from .re_weighting import Re_Weighting_Strategy


class RAG:

    def __init__(self,
                 model_name: str = "openai",
                 retriever_path: str = "",
                 reranker_path: str = "",
                 normalize_L2: bool = False,
                 distance_strategy="EUCLIDEAN_DISTANCE",
                 whether_print_info: bool = True,
                 whether_my_decoding: bool = False,
                 whether_re_weighting: bool = False,
                 layers_to_be_modified: list = []):
        """model_name: openai or llama2_7b_chat or llama2_13b_chat or llama3 or mistral or gemma or qwen or cag"""
        self.model_name = model_name
        self.retriever_path = retriever_path
        self.reranker_path = reranker_path
        self.normalize_L2 = normalize_L2
        self.distance_strategy = distance_strategy
        self.whether_print_info = whether_print_info
        self.whether_my_decoding = whether_my_decoding
        self.whether_re_weighting = whether_re_weighting
        self.print_colors = [Fore.YELLOW, Fore.GREEN]
        self.current_color_index = 0 
        if self.retriever_path != "":
            self.init_retriever()
        if self.reranker_path != "":
            self.init_reranker()
        if self.model_name != "openai" and self.whether_re_weighting == True:
            self.init_re_weighting_model(layers_to_be_modified=layers_to_be_modified)
        elif self.model_name != "openai" and self.whether_my_decoding == True:
            self.init_chat_model_with_my_decoding()
        elif self.model_name != "openai" and self.whether_my_decoding == False:
            self.init_chat_model()
        print("RAG init success")

    def print_info(self, info: str = ''):
        if self.whether_print_info:
            current_color = self.print_colors[self.current_color_index]
            print(f"{current_color}{info}{Style.RESET_ALL}")
            self.current_color_index = (self.current_color_index + 1) % len(self.print_colors)

    def init_retriever(self):
        if self.retriever_path.find("bge") != -1:
            model_name = "bge-large-en-v1.5"
        elif self.retriever_path.find("contriever") != -1:
            model_name = "contriever-msmarco"
        embeddings_model = HuggingFaceEmbeddings(model_name=f"./model/{model_name}", model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True})
        if pkg_version.parse(version('langchain')) > pkg_version.parse('0.0.285'):
            self.vector_store = FAISS.load_local(f"{self.retriever_path}/", embeddings=embeddings_model, normalize_L2=self.normalize_L2, distance_strategy=self.distance_strategy, allow_dangerous_deserialization=True)
            self.print_info(f"langchain version: {version('langchain')}, use allow_dangerous_deserialization=True")
        else:
            self.vector_store = FAISS.load_local(f"{self.retriever_path}/", embeddings=embeddings_model, normalize_L2=self.normalize_L2, distance_strategy=self.distance_strategy)
            self.print_info(f"langchain version: {version('langchain')}, use allow_dangerous_deserialization=False")

    def init_reranker(self):
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_path)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_path, device_map="auto")
        self.reranker_model.eval()

    def init_chat_model(self):
        if self.model_name == 'llama2_13b_chat':
            model_name = "Llama-2-13b-chat-hf"
            self.bad_words_ids = [[13]]
        elif self.model_name == 'llama3':
            model_name = "Meta-Llama-3-8B-Instruct"
            self.bad_words_ids = []
        elif self.model_name == 'qwen':
            model_name = 'Qwen1.5-7B-Chat'
            self.bad_words_ids = []
        elif self.model_name == 'cag':
            model_name = 'CAG-13b'
            self.bad_words_ids = []
        self.chat_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.model_name == 'cag':
            self.chat_model_pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.chat_model_pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                torch_dtype="auto",
                device_map="auto",
            )

    def init_re_weighting_model(self, layers_to_be_modified):
        if self.model_name == 'llama2_13b_chat':
            model_name = "Llama-2-13b-chat-hf"
            self.bad_words_ids = [[13]]
        elif self.model_name == 'llama3':
            model_name = "Meta-Llama-3-8B-Instruct"
        elif self.model_name == 'qwen':
            model_name = 'Qwen1.5-7B-Chat'
            self.bad_words_ids = []
        self.chat_model_re_weighting = Re_Weighting_Strategy(model_name, layers_to_be_modified=layers_to_be_modified, bad_words_ids=self.bad_words_ids)

    def run_retriever(self, question: str = '', topk: int = 1):
        docs = self.vector_store.similarity_search(question, k=topk)
        paras = [doc.page_content for doc in docs]
        return paras

    def run_reranker(self, question: str = '', paras: list = [], topk: int = 1):
        pairs = [[question, para] for para in paras]
        with torch.no_grad():
            inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float().to('cpu')
            # print(scores)
            topk_indices = torch.argsort(scores, descending=True)[:topk]
            # Retrieve paragraphs corresponding to topk indices
            topk_paras = [paras[idx] for idx in topk_indices]
            return topk_paras

    def run_RAG(self, question: str = '', paras: list = [], scores: list = [], type: str = 'with_contexts'):
        if self.whether_re_weighting == True:
            prompt, answer = self.chat_model_re_weighting.run_RAG_with_attention_weighting(question=question, paras=paras, scores=scores)
            return prompt, answer
        else:
            # if paras == []:
            #     paras = self.run_retriever(question)
            if 'llama2' in self.model_name and type == 'with_contexts_score':
                type = 'with_contexts_score_llama2'
            prompt = get_prompt(context=paras, question=question, type=type, scores=scores)
            answer = self.get_LLM_answer(prompt)
            return prompt, answer

    def run_RAG_without_contexts(self, question: str = ''):
        prompt = get_prompt(question=question, type='without_contexts')
        answer = self.get_LLM_answer(prompt)
        return prompt, answer

    def get_LLM_answer(self, prompt: str = ''):
        if self.model_name == "openai":
            result = askChatGPT(prompt)
        else:
            result = self.askChatModel(prompt)
        return result

    def get_truthful_judge(self, para: str = '', question: str = ''):
        cnt = 0
        prompt = get_prompt_truthful_judge(para, question)
        while cnt < 2:
            result = self.get_LLM_answer(prompt)
            match = re.search(r"Credibility Score: (\d+)", result, re.IGNORECASE)
            if match:
                score = match.group(1)
                return score
            cnt += 1
        print("Can't get the Credibility Score")
        score = 1
        return score

    def askChatModel(self, prompt: str = ""):
        para_dict = {
            "do_sample": False,
            "max_new_tokens": 100,
            "add_special_tokens": True,
        }
        if self.model_name == "llama3":
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
            prompt = self.chat_model_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            para_dict["eos_token_id"] = [self.chat_model_pipeline.tokenizer.eos_token_id, self.chat_model_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        elif self.model_name == 'qwen':
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ]
            para_dict["add_special_tokens"] = False
            prompt = self.chat_model_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.bad_words_ids:
            para_dict["bad_words_ids"] = self.bad_words_ids
        if self.whether_my_decoding == False:
            sequences = self.chat_model_pipeline(prompt, **para_dict)
            for seq in sequences:
                generated_text = seq['generated_text']
                prompt_end_index = generated_text.find(prompt) + len(prompt)
                output = generated_text[prompt_end_index:]
        else:
            output = self.chat_model_with_my_decoding.gen_outputs(prompt)
        return output
