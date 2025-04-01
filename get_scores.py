import argparse
import json
import math
import os
import requests
import torch
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Import SePer related modules
from seper.calculate import calculate_uncertainty_soft_batch, create_collate_fn
from seper.uncertainty_measures.semantic_entropy import EntailmentDeberta
from seper.calculate import process_item_for_seper

# Import Semantic Entropy related modules
from seper.uncertainty_measures.semantic_entropy import (
    get_semantic_ids, logsumexp_by_id, predictive_entropy_rao
)


class RerankerModel:
    """Handles reranking for different types of reranker models."""

    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

        # Load the model based on the model name
        self._load_model()

    def _load_model(self):
        """Load the appropriate reranker model based on model name."""
        tqdm.write(f"Loading reranker model: {self.model_name}")

        if 'layerwise' in self.model_name.lower():
            # LLM-based layerwise reranker
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            self.model = self.model.to(self.device)
            self.model_type = "llm-layerwise"

        elif any(llm_marker in self.model_name.lower() for llm_marker in ['llama', 'mistral', 'gemma']):
            # LLM-based reranker
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)[
                'input_ids'][0]
            self.model_type = "llm"

        else:
            # Standard reranker
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name)
            self.model = self.model.to(self.device)
            self.model_type = "standard"

        self.model.eval()
        tqdm.write(
            f"Reranker model loaded: {self.model_name} (Type: {self.model_type})")

    def get_inputs_llm(self, pairs, prompt=None, max_length=1024):
        """Get inputs for LLM-based rerankers."""
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = self.tokenizer(prompt,
                                       return_tensors=None,
                                       add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(f'A: {query}',
                                          return_tensors=None,
                                          add_special_tokens=False,
                                          max_length=max_length * 3 // 4,
                                          truncation=True)
            passage_inputs = self.tokenizer(f'B: {passage}',
                                            return_tensors=None,
                                            add_special_tokens=False,
                                            max_length=max_length,
                                            truncation=True)
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        ).to(self.device)

    def score(self, query: str, passages: List[str]) -> List[float]:
        """Score query-passage pairs based on relevance."""
        pairs = [[query, passage] for passage in passages]

        with torch.no_grad():
            if self.model_type == "standard":
                inputs = self.tokenizer(
                    pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
                scores = self.model(
                    **inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy().tolist()

            elif self.model_type == "llm":
                inputs = self.get_inputs_llm(pairs)
                scores = self.model(
                    **inputs, return_dict=True).logits[:, -1, self.yes_loc].view(-1, ).float().cpu().numpy().tolist()

            elif self.model_type == "llm-layerwise":
                inputs = self.get_inputs_llm(pairs)
                all_scores = self.model(
                    **inputs, return_dict=True, cutoff_layers=[28])
                # Extract scores from the first layer in the cutoff_layers
                scores = [score[:, -1].view(-1, ).float().cpu().numpy().tolist()
                          for score in all_scores[0]][0]

        return scores


class APIClient:
    """Handles API interactions with the LLM server."""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}

    def make_request(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        num_logprobs: int
    ) -> Dict[str, Any]:
        """Make a request to the API with the given parameters."""
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": num_logprobs
        }

        try:
            response = requests.post(
                self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            tqdm.write(f"API request failed: {e}")
            return {"choices": [{"message": {"content": "API request failed"}, "logprobs": {"content": []}}]}


class MetricsCalculator:
    """Calculates various metrics from model responses."""

    @staticmethod
    def extract_logprobs(response_data: Dict[str, Any]) -> List[float]:
        """Extract logprobs from the API response."""
        try:
            tokens_data = response_data.get("choices", [{}])[0].get(
                "logprobs", {}).get("content", [])
            return [token_info.get("logprob", 0) for token_info in tokens_data]
        except Exception as e:
            tqdm.write(f"Error extracting logprobs: {e}")
            return [0.0]

    @staticmethod
    def calculate_entropy(logprobs: List[float]) -> float:
        """Calculate entropy from logprobs."""
        if not logprobs:
            return 0.0
        probs = [math.exp(lp) for lp in logprobs]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return entropy / len(logprobs)

    @staticmethod
    def calculate_perplexity(logprobs: List[float]) -> float:
        """Calculate perplexity from logprobs."""
        if not logprobs:
            return float('inf')
        avg_logprob = sum(logprobs) / len(logprobs)
        return math.exp(-avg_logprob)

    def calculate_metrics(self, logprobs: List[float]) -> Dict[str, float]:
        """Calculate all metrics from logprobs."""
        return {
            "entropy": self.calculate_entropy(logprobs),
            "perplexity": self.calculate_perplexity(logprobs)
        }


class DocumentProcessor:
    """Handles document formatting and message preparation."""

    @staticmethod
    def format_documents(documents: List[Dict[str, Any]], start_index: int = 1) -> str:
        """Format a list of documents into a single string.

        Args:
            documents: List of document dictionaries
            start_index: Starting index for document numbering (default: 1)
        """
        doc_text = ""
        for i, doc in enumerate(documents, start_index):
            content = doc.get('contents', '')
            doc_text += f"Doc {i}: {content}\n"
        return doc_text

    @staticmethod
    def prepare_chat_messages(question: str, doc_text: str = None) -> List[Dict[str, str]]:
        """Prepare chat messages with or without document context."""
        if doc_text:
            return [
                {
                    "role": "system",
                    "content": "Answer the following question based on the given document. Only give me the answer and do not output any other words."
                },
                {
                    "role": "user",
                    "content": f"The following are given documents: {doc_text}\n\nQuestion: {question}"
                }
            ]
        else:
            return [
                {
                    "role": "system",
                    "content": "Answer the following question. Only give me the answer and do not output any other words."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}"
                }
            ]


class SePerCalculator:
    """Handles SePer-specific calculations."""

    def __init__(self, device: str = "cuda:0", entailment_model=None):
        self.device = device
        self.entailment_model = entailment_model
        # 只有在未提供模型的情况下才初始化新模型
        if self.entailment_model is None:
            self.initialize_model()

    def initialize_model(self):
        """Initialize the entailment model for SePer."""
        tqdm.write("Initializing SePer Entailment Model...")
        self.entailment_model = EntailmentDeberta(device=self.device)
        self.entailment_model.model.eval()
        tqdm.write("SePer Entailment Model initialization complete")

    def prepare_input(
        self,
        question: str,
        answers: List[str],
        context: str,
        generations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare input for SePer calculation."""
        result = {
            "question": question,
            "context": context,
            "answers": answers,
            "responses": [{"response": gen["text"], "log_likelihood": gen["logprob"]} for gen in generations]
        }

        return process_item_for_seper(result)

    def calculate_seper(
        self,
        input_data: Dict[str, Any],
        computation_chunk_size: int = 8
    ) -> float:
        """Calculate SePer for a single input."""
        keys = ['question', 'response_text', 'answers',
                'likelihood', 'context_label', 'log_liks_agg', 'context']
        seper_collate_fn = create_collate_fn(keys)

        with torch.no_grad():
            seper_input = seper_collate_fn([input_data])
            results = calculate_uncertainty_soft_batch(
                seper_input, self.entailment_model, computation_chunk_size
            )
            return float(results[0])

    def calculate_reduction(
        self,
        baseline_data: Dict[str, Any],
        context_data: Dict[str, Any],
        computation_chunk_size: int = 8,
        baseline_seper: float = None  # 新增参数，允许传入预计算的基线SePer
    ) -> Tuple[float, float, float]:
        """Calculate the difference in SePer between baseline and context."""

        keys = ['question', 'response_text', 'answers',
                'likelihood', 'context_label', 'log_liks_agg', 'context']
        seper_collate_fn = create_collate_fn(keys)

        with torch.no_grad():
            # 仅在未提供预计算值时计算基线SePer
            if baseline_seper is None:
                baseline_input = seper_collate_fn([baseline_data])
                baseline_results = calculate_uncertainty_soft_batch(
                    baseline_input, self.entailment_model, computation_chunk_size
                )
                baseline_seper = float(baseline_results[0])

            # 计算context的SePer
            context_input = seper_collate_fn([context_data])
            context_results = calculate_uncertainty_soft_batch(
                context_input, self.entailment_model, computation_chunk_size
            )
            seper_context = float(context_results[0])

            # 计算reduction
            seper_reduction = seper_context - baseline_seper

        return seper_context, baseline_seper, seper_reduction


class SemanticEntropyCalculator:
    """Handles Semantic Entropy calculations."""

    def __init__(self, device: str = "cuda:0", entailment_model=None):
        self.device = device
        self.entailment_model = entailment_model
        # 只有在未提供模型的情况下才初始化新模型
        if self.entailment_model is None:
            self.initialize_model()

    def initialize_model(self):
        """Initialize the entailment model for Semantic Entropy."""
        tqdm.write("Initializing Semantic Entropy Entailment Model...")
        self.entailment_model = EntailmentDeberta(device=self.device)
        self.entailment_model.model.eval()
        tqdm.write("Semantic Entropy Entailment Model initialization complete")

    def calculate_semantic_entropy(
        self,
        question: str,
        answers: List[str],
        context: str,
        generations: List[Dict[str, Any]],
        strict_entailment: bool = False
    ) -> float:
        """Calculate semantic entropy from a list of generated responses."""

        # 提取回答文本和对数似然
        responses = [gen["text"] for gen in generations]
        log_liks = [gen["logprob"] for gen in generations]

        # 创建example格式
        example = {
            "question": question,
            "context": context,
            "reference": {"answers": {"text": answers}}
        }

        # tqdm.write(f"{question}\n {len(context.split('Doc'))-1}\n {answers}")

        # 计算语义ID (对回答进行语义聚类)
        semantic_ids = get_semantic_ids(
            responses,
            model=self.entailment_model,
            strict_entailment=strict_entailment,
            example=example
        )

        # 计算语义熵
        log_likelihood_per_semantic_id = logsumexp_by_id(
            semantic_ids, log_liks, agg='sum_normalized')
        semantic_entropy = float(predictive_entropy_rao(
            log_likelihood_per_semantic_id))

        return semantic_entropy

    def calculate_reduction(
        self,
        question: str,
        golden_answers: List[str],
        context: str,
        baseline_generations: List[Dict[str, Any]],
        context_generations: List[Dict[str, Any]],
        strict_entailment: bool = False,
        baseline_entropy: float = None  # 新增参数，允许传入预计算的基线熵
    ) -> Tuple[float, float, float]:
        """Calculate the difference in Semantic Entropy between baseline and context."""

        # 只有在没有提供预计算基线熵时才计算
        if baseline_entropy is None:
            baseline_entropy = self.calculate_semantic_entropy(
                question=question,
                answers=golden_answers,
                context="",
                generations=baseline_generations,
                strict_entailment=strict_entailment
            )

        context_entropy = self.calculate_semantic_entropy(
            question=question,
            answers=golden_answers,
            context=context,
            generations=context_generations,
            strict_entailment=strict_entailment
        )

        # 计算减少量
        entropy_reduction = baseline_entropy - context_entropy

        return context_entropy, baseline_entropy, entropy_reduction


class RAGEvaluator:
    """Main class for evaluating RAG performance."""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        num_logprobs: int = 5,
        use_seper: bool = False,
        use_semantic_entropy: bool = False,
        model_device: str = "cuda:0",
        num_generations: int = 10,
        generation_temperature: float = 1.0,
        generation_max_tokens: int = 128,
        computation_chunk_size: int = 8,
        strict_entailment: bool = False,
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ):
        self.api_client = APIClient(api_url)
        self.metrics_calculator = MetricsCalculator()
        self.doc_processor = DocumentProcessor()

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_logprobs = num_logprobs

        # 初始化共享的EntailmentDeberta模型
        self.shared_entailment_model = None
        if use_seper or use_semantic_entropy:
            tqdm.write("Initializing shared EntailmentDeberta model...")
            from seper.uncertainty_measures.semantic_entropy import EntailmentDeberta
            self.shared_entailment_model = EntailmentDeberta(
                device=model_device)
            self.shared_entailment_model.model.eval()
            tqdm.write("Shared EntailmentDeberta model initialization complete")

        # SePer相关设置
        self.use_seper = use_seper
        if use_seper:
            self.seper_calculator = SePerCalculator(
                device=model_device,
                entailment_model=self.shared_entailment_model
            )

        # 语义熵相关设置
        self.use_semantic_entropy = use_semantic_entropy
        if use_semantic_entropy:
            self.semantic_entropy_calculator = SemanticEntropyCalculator(
                device=model_device,
                entailment_model=self.shared_entailment_model
            )

        # Reranker相关设置
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        if use_reranker:
            tqdm.write(f"Initializing reranker model: {reranker_model}")
            self.reranker = RerankerModel(reranker_model, device=model_device)

        # 共享的生成参数
        self.num_generations = num_generations
        self.generation_temperature = generation_temperature
        self.generation_max_tokens = generation_max_tokens
        self.computation_chunk_size = computation_chunk_size
        self.strict_entailment = strict_entailment
        self.model_device = model_device

    def _get_response(self, question: str, doc_text: str = None) -> Dict[str, Any]:
        """Get a response from the API with metrics."""
        messages = self.doc_processor.prepare_chat_messages(question, doc_text)

        response = self.api_client.make_request(
            model_name=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            num_logprobs=self.num_logprobs
        )

        logprobs = self.metrics_calculator.extract_logprobs(response)
        metrics = self.metrics_calculator.calculate_metrics(logprobs)

        return {
            "response": response["choices"][0]["message"]["content"],
            "metrics": metrics,
            "logprobs": logprobs
        }

    def _generate_multiple_answers(
        self,
        question: str,
        context: str
    ) -> List[Dict[str, Any]]:
        """Generate multiple answers for uncertainty metrics calculation."""
        results = []
        messages = self.doc_processor.prepare_chat_messages(question, context)

        for _ in range(self.num_generations):
            response = self.api_client.make_request(
                model_name=self.model_name,
                messages=messages,
                temperature=self.generation_temperature,
                max_tokens=self.generation_max_tokens,
                num_logprobs=5
            )

            if "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"]
                logprobs = self.metrics_calculator.extract_logprobs(response)
                # Length Normalized
                avg_logprob = sum(logprobs)/len(logprobs) if logprobs else 0.0

                results.append({
                    "text": generated_text,
                    "logprob": avg_logprob
                })

        return results

    def _calculate_reranker_scores(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        batch_size: int = 8
    ) -> List[float]:
        """Calculate reranker scores for the retrieved documents."""
        passages = [doc.get('contents', '') for doc in retrieved_docs]

        if not self.use_reranker or not passages:
            return [0.0] * len(retrieved_docs)

        try:
            # Process in batches to avoid memory issues
            scores = []
            for i in range(0, len(passages), batch_size):
                batch_passages = passages[i:i+batch_size]
                batch_scores = self.reranker.score(question, batch_passages)
                scores.extend(batch_scores)

            return scores
        except Exception as e:
            tqdm.write(f"Error calculating reranker scores: {e}")
            import traceback
            traceback.print_exc()
            return [0.0] * len(retrieved_docs)

    def _process_individual_docs(
        self,
        question: str,
        naive_metrics: Dict[str, float],
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process each individual document and calculate metrics."""
        results = []

        for i, doc in enumerate(retrieved_docs):
            # 使用文档的原始索引(i+1)而不是总是从1开始
            single_doc_text = self.doc_processor.format_documents(
                [doc], start_index=i+1)
            single_doc_result = self._get_response(question, single_doc_text)

            single_doc_utility = {
                "entropy_reduction": naive_metrics["entropy"] - single_doc_result["metrics"]["entropy"],
                "perplexity_reduction": naive_metrics["perplexity"] - single_doc_result["metrics"]["perplexity"],
                "retriever_score": doc.get('score', 0.0)  # 添加retriever得分
            }

            doc_result = {
                "doc_index": i,
                "doc_id": doc.get('id', f"doc_{i}"),
                "doc_score": doc.get('score', 0.0),
                "doc_content": doc.get('contents', ''),
                "response": single_doc_result["response"],
                "metrics": single_doc_result["metrics"],
                "utility": single_doc_utility
            }

            results.append(doc_result)

        return results

    def _calculate_seper_metrics(
        self,
        question: str,
        golden_answers: List[str],
        doc_text: str,
        retrieved_docs: List[Dict[str, Any]],
        baseline_generations=None,
        all_docs_generations=None,
        doc_generations=None
    ) -> Optional[Dict[str, Any]]:
        """Calculate SePer metrics using provided or new generations."""
        try:
            # 使用提供的生成结果或生成新的
            if baseline_generations is None:
                baseline_generations = self._generate_multiple_answers(
                    question, "")

            if all_docs_generations is None:
                all_docs_generations = self._generate_multiple_answers(
                    question, doc_text)

            # 准备基线数据
            baseline_data = self.seper_calculator.prepare_input(
                question=question,
                answers=golden_answers,
                context="",
                generations=baseline_generations
            )

            # 提前计算基线SePer，只计算一次
            baseline_seper = self.seper_calculator.calculate_seper(
                input_data=baseline_data,
                computation_chunk_size=self.computation_chunk_size
            )

            # 准备所有文档数据
            all_docs_data = self.seper_calculator.prepare_input(
                question=question,
                answers=golden_answers,
                context=doc_text,
                generations=all_docs_generations
            )

            # 使用预计算的基线SePer计算所有文档的reduction
            all_docs_seper, _, all_docs_seper_reduction = self.seper_calculator.calculate_reduction(
                baseline_data=baseline_data,
                context_data=all_docs_data,
                computation_chunk_size=self.computation_chunk_size,
                baseline_seper=baseline_seper  # 传入预计算的基线SePer
            )

            # Calculate individual document SePer
            individual_doc_results = []

            for i, doc in enumerate(retrieved_docs):
                # 使用文档的原始索引(i+1)
                single_doc_text = self.doc_processor.format_documents(
                    [doc], start_index=i+1)

                # 使用提供的生成结果或生成新的
                single_doc_generations = None
                if doc_generations is not None and i < len(doc_generations):
                    single_doc_generations = doc_generations[i]

                if single_doc_generations is None:
                    single_doc_generations = self._generate_multiple_answers(
                        question, single_doc_text)

                single_doc_data = self.seper_calculator.prepare_input(
                    question=question,
                    answers=golden_answers,
                    context=single_doc_text,
                    generations=single_doc_generations
                )

                # 使用预计算的基线SePer
                single_doc_seper, _, single_doc_seper_reduction = self.seper_calculator.calculate_reduction(
                    baseline_data=baseline_data,
                    context_data=single_doc_data,
                    computation_chunk_size=self.computation_chunk_size,
                    baseline_seper=baseline_seper  # 传入预计算的基线SePer
                )

                individual_doc_results.append({
                    "doc_index": i,
                    "doc_id": doc.get('id', f"doc_{i}"),
                    "seper": single_doc_seper,
                    "seper_reduction": single_doc_seper_reduction,
                    "generations": single_doc_generations  # 保存生成结果以便共享
                })

            return {
                "baseline_seper": baseline_seper,
                "all_docs_seper": all_docs_seper,
                "all_docs_seper_reduction": all_docs_seper_reduction,
                "individual_docs": individual_doc_results,
                "baseline_generations": baseline_generations,  # 保存生成结果以便共享
                "all_docs_generations": all_docs_generations  # 保存生成结果以便共享
            }

        except Exception as e:
            tqdm.write(f"Error calculating SePer metrics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_semantic_entropy_metrics(
        self,
        question: str,
        golden_answers: List[str],
        doc_text: str,
        retrieved_docs: List[Dict[str, Any]],
        baseline_generations=None,
        all_docs_generations=None,
        doc_generations=None
    ) -> Optional[Dict[str, Any]]:
        """Calculate Semantic Entropy metrics using provided or new generations."""
        try:
            # 使用提供的生成结果或生成新的
            if baseline_generations is None:
                baseline_generations = self._generate_multiple_answers(
                    question, "")

            if all_docs_generations is None:
                all_docs_generations = self._generate_multiple_answers(
                    question, doc_text)

            # 首先计算基线熵，只计算一次
            baseline_entropy = self.semantic_entropy_calculator.calculate_semantic_entropy(
                question=question,
                answers=golden_answers,
                context="",
                generations=baseline_generations,
                strict_entailment=self.strict_entailment
            )

            # Calculate semantic entropy for all docs and get reduction
            all_docs_entropy = self.semantic_entropy_calculator.calculate_semantic_entropy(
                question=question,
                answers=golden_answers,
                context=doc_text,
                generations=all_docs_generations,
                strict_entailment=self.strict_entailment
            )

            all_docs_entropy_reduction = baseline_entropy - all_docs_entropy

            # Calculate individual document semantic entropy
            individual_doc_results = []

            for i, doc in enumerate(retrieved_docs):
                # 使用文档的原始索引(i+1)
                single_doc_text = self.doc_processor.format_documents(
                    [doc], start_index=i+1)

                # 使用提供的生成结果或生成新的
                single_doc_generations = None
                if doc_generations is not None and i < len(doc_generations):
                    single_doc_generations = doc_generations[i]

                if single_doc_generations is None:
                    single_doc_generations = self._generate_multiple_answers(
                        question, single_doc_text)

                # 使用预计算的基线熵计算reduction
                single_doc_entropy = self.semantic_entropy_calculator.calculate_semantic_entropy(
                    question=question,
                    answers=golden_answers,
                    context=single_doc_text,
                    generations=single_doc_generations,
                    strict_entailment=self.strict_entailment
                )

                single_doc_entropy_reduction = baseline_entropy - single_doc_entropy

                individual_doc_results.append({
                    "doc_index": i,
                    "doc_id": doc.get('id', f"doc_{i}"),
                    "semantic_entropy": single_doc_entropy,
                    "semantic_entropy_reduction": single_doc_entropy_reduction
                })

            return {
                "baseline_semantic_entropy": baseline_entropy,
                "all_docs_semantic_entropy": all_docs_entropy,
                "all_docs_semantic_entropy_reduction": all_docs_entropy_reduction,
                "individual_docs": individual_doc_results
            }

        except Exception as e:
            tqdm.write(f"Error calculating Semantic Entropy metrics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with its retrieved documents."""
        question = item['question']
        golden_answers = item['golden_answers']
        retrieved_docs = item['output']['retrieval_result']

        # Get naive (no documents) response
        naive_result = self._get_response(question)

        # Get RAG (all documents) response
        doc_text = self.doc_processor.format_documents(retrieved_docs)
        rag_result = self._get_response(question, doc_text)

        # Calculate utility
        utility = {
            "entropy_reduction": naive_result["metrics"]["entropy"] - rag_result["metrics"]["entropy"],
            "perplexity_reduction": naive_result["metrics"]["perplexity"] - rag_result["metrics"]["perplexity"]
        }

        # Process individual documents (now includes retriever_score)
        individual_doc_results = self._process_individual_docs(
            question, naive_result["metrics"], retrieved_docs)

        # Calculate reranker scores if enabled
        if self.use_reranker:
            reranker_scores = self._calculate_reranker_scores(
                question, retrieved_docs)

            # Add reranker scores to individual document results
            for i, doc_result in enumerate(individual_doc_results):
                if i < len(reranker_scores):
                    doc_result["utility"]["reranker_score"] = reranker_scores[i]

        # 准备生成共享变量
        baseline_generations = None
        all_docs_generations = None
        doc_generations = None

        # 如果同时启用了两个指标，先生成一次共享结果
        if self.use_seper and self.use_semantic_entropy:
            tqdm.write(
                "Generating shared samples for both SePer and Semantic Entropy...")
            baseline_generations = self._generate_multiple_answers(
                question, "")
            all_docs_generations = self._generate_multiple_answers(
                question, doc_text)

            doc_generations = []
            for i, doc in enumerate(retrieved_docs):
                # 使用文档的原始索引(i+1)
                single_doc_text = self.doc_processor.format_documents(
                    [doc], start_index=i+1)
                single_doc_generations = self._generate_multiple_answers(
                    question, single_doc_text)
                doc_generations.append(single_doc_generations)

        # Calculate SePer metrics if enabled
        if self.use_seper:
            seper_results = self._calculate_seper_metrics(
                question, golden_answers, doc_text, retrieved_docs,
                baseline_generations, all_docs_generations, doc_generations
            )

            if seper_results:
                naive_result["metrics"]["seper"] = seper_results["baseline_seper"]
                rag_result["metrics"]["seper"] = seper_results["all_docs_seper"]
                utility["seper_reduction"] = seper_results["all_docs_seper_reduction"]

                # 如果需要为语义熵共享生成结果
                if self.use_semantic_entropy and baseline_generations is None:
                    baseline_generations = seper_results["baseline_generations"]
                    all_docs_generations = seper_results["all_docs_generations"]
                    doc_generations = [doc_result.get(
                        "generations") for doc_result in seper_results["individual_docs"]]

                # Update individual document results with SePer metrics
                for i, doc_result in enumerate(individual_doc_results):
                    if i < len(seper_results["individual_docs"]):
                        doc_result["metrics"]["seper"] = seper_results["individual_docs"][i]["seper"]
                        doc_result["utility"]["seper_reduction"] = seper_results["individual_docs"][i]["seper_reduction"]

        # Calculate Semantic Entropy metrics if enabled
        if self.use_semantic_entropy:
            semantic_entropy_results = self._calculate_semantic_entropy_metrics(
                question, golden_answers, doc_text, retrieved_docs,
                baseline_generations, all_docs_generations, doc_generations
            )

            if semantic_entropy_results:
                naive_result["metrics"]["semantic_entropy"] = semantic_entropy_results["baseline_semantic_entropy"]
                rag_result["metrics"]["semantic_entropy"] = semantic_entropy_results["all_docs_semantic_entropy"]
                utility["semantic_entropy_reduction"] = semantic_entropy_results["all_docs_semantic_entropy_reduction"]

                # Update individual document results with Semantic Entropy metrics
                for i, doc_result in enumerate(individual_doc_results):
                    if i < len(semantic_entropy_results["individual_docs"]):
                        doc_result["metrics"]["semantic_entropy"] = semantic_entropy_results["individual_docs"][i]["semantic_entropy"]
                        doc_result["utility"]["semantic_entropy_reduction"] = semantic_entropy_results[
                            "individual_docs"][i]["semantic_entropy_reduction"]

        # Build final result
        result = {
            "id": item['id'],
            "question": question,
            "golden_answers": golden_answers,
            "naive_response": naive_result["response"],
            "rag_response": rag_result["response"],
            "naive_metrics": naive_result["metrics"],
            "rag_metrics": rag_result["metrics"],
            "utility": utility,
            "individual_doc_results": individual_doc_results
        }

        return result

    def evaluate_dataset(
        self,
        dataset_path: str,
        output_path: str,
        max_samples: Optional[int] = None
    ) -> None:
        """Evaluate the entire dataset and save results."""
        # 加载数据集
        data = load_dataset(dataset_path)
        if max_samples:
            data = data[:max_samples]

        results = []

        # 处理每个问题
        for idx, item in tqdm(enumerate(data), total=len(data), desc="Evaluating"):
            tqdm.write(
                f"Processing question {idx+1}/{len(data)}: {item['question'][:50]}...")

            result = self.evaluate_question(item)
            results.append(result)

            # 保存中间结果
            if (idx + 1) % 10 == 0:
                temp_output_path = f"{os.path.splitext(output_path)[0]}_temp_{idx+1}.json"
                tqdm.write(
                    f"Saving intermediate results to {temp_output_path}")
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存最终结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        tqdm.write(f"Evaluation complete, results saved to {output_path}")


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG performance")
    parser.add_argument("--dataset_path", type=str,
                        default="data/nq_2024_11_07_12_14_naive/intermediate_data.json")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_path", type=str,
                        default="data/nq_2024_11_07_12_14_naive/results/rag_evaluation_results.json")
    parser.add_argument("--api_url", type=str,
                        default="http://localhost:40000/v1/chat/completions")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_logprobs", type=int, default=20)

    # 通用生成设置
    parser.add_argument("--model_device", type=str, default="cuda:0",
                        help="Device for model calculations")
    parser.add_argument("--num_generations", type=int, default=10,
                        help="Number of generations for metrics calculation")
    parser.add_argument("--generation_temperature", type=float, default=1.0,
                        help="Temperature for generations")
    parser.add_argument("--generation_max_tokens", type=int, default=128,
                        help="Max tokens for generations")
    parser.add_argument("--computation_chunk_size", type=int, default=8,
                        help="Computation chunk size for batched operations")

    # SePer related parameters
    parser.add_argument("--use_seper", action="store_true",
                        help="Use SePer for evaluation")

    # Semantic Entropy related parameters
    parser.add_argument("--use_semantic_entropy", action="store_true",
                        help="Use Semantic Entropy for evaluation")
    parser.add_argument("--strict_entailment", action="store_true",
                        help="Use strict entailment for semantic clustering")

    # Reranker related parameters
    parser.add_argument("--use_reranker", action="store_true",
                        help="Use reranker for evaluation")
    parser.add_argument("--reranker_model", type=str,
                        default="BAAI/bge-reranker-v2-m3",  # Normal reranker
                        # "BAAI/bge-reranker-v2-gemma" # LLM-based reranker
                        # "BAAI/bge-reranker-v2-minicpm-layerwise" # LLM-based layerwise reranker
                        help="Reranker model name")

    args = parser.parse_args()

    evaluator = RAGEvaluator(
        api_url=args.api_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_logprobs=args.num_logprobs,
        use_seper=args.use_seper,
        use_semantic_entropy=args.use_semantic_entropy,
        model_device=args.model_device,
        num_generations=args.num_generations,
        generation_temperature=args.generation_temperature,
        generation_max_tokens=args.generation_max_tokens,
        computation_chunk_size=args.computation_chunk_size,
        strict_entailment=args.strict_entailment,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model
    )

    evaluator.evaluate_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
