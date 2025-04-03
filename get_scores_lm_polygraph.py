import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

# Import vLLM instead of transformers
from vllm import LLM, SamplingParams

# Import lm_polygraph with vLLM support
from lm_polygraph.model_adapters import WhiteboxModelvLLM
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.estimators import (
    MaximumSequenceProbability,
    Perplexity,
    MeanTokenEntropy,
    MonteCarloSequenceEntropy,
    MonteCarloNormalizedSequenceEntropy,
    RenyiNeg,
    FisherRao,
    SemanticEntropy,
    ClaimConditionedProbability,
    TokenSAR,
    SentenceSAR,
    SAR,
    EigenScore,
    PTrue,
    NumSemSets,
    EigValLaplacian,
    DegMat,
    Eccentricity,
    LexicalSimilarity,
    KernelLanguageEntropy,
    LUQ
)
from lm_polygraph.stat_calculators.greedy_alternatives_nli import GreedyAlternativesNLICalculator
from lm_polygraph.stat_calculators.cross_encoder_similarity import CrossEncoderSimilarityMatrixCalculator
from lm_polygraph.stat_calculators.semantic_matrix import SemanticMatrixCalculator
from lm_polygraph.stat_calculators.semantic_classes import SemanticClassesCalculator
from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator
from lm_polygraph.stat_calculators.sample import SamplingGenerationCalculator
from lm_polygraph.stat_calculators.entropy import EntropyCalculator
from lm_polygraph.stat_calculators.prompt import PromptCalculator
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
import time
import functools
from typing import List, Dict, Any, Callable

def time_calculator(func: Callable) -> Callable:
    """测量计算器方法执行时间的装饰器。"""
    @functools.wraps(func)
    def wrapper(self, deps: Dict[str, Any], texts: List[str], model: Any) -> Dict[str, Any]:
        start_time = time.time()
        result = func(self, deps, texts, model)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        # Handle infinity and NaN values
        elif obj == float('inf'):
            return "Infinity"
        elif obj == float('-inf'):
            return "-Infinity"
        elif np.isnan(obj):
            return "NaN"
        return super().default(obj)


def json_infinity_decoder(obj):
    """Custom object hook for json.loads to convert string representations of infinity back to float."""
    for key, value in obj.items():
        if isinstance(value, str):
            if value == "Infinity":
                obj[key] = float('inf')
            elif value == "-Infinity":
                obj[key] = float('-inf')
            elif value == "NaN":
                obj[key] = float('nan')
    return obj


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from a JSON file with support for Infinity values."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f, object_hook=json_infinity_decoder)
    return data


def save_json_with_infinity(data, file_path: str) -> None:
    """Save data to a JSON file with support for Infinity values."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

class VLLMModel:
    """Handles direct interaction with vLLM models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_tokens: int = 128,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.7,
    ):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization

        # Set CUDA_VISIBLE_DEVICES based on the device
        if "cuda:" in device:
            device_id = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        
        # Load the model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the vLLM model and tokenizer."""
        tqdm.write(f"Loading vLLM model: {self.model_name}")
        
        # Initialize vLLM
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization
        )
        
        # Get tokenizer from vLLM
        self.tokenizer = self.llm.get_tokenizer()
        
        # Create sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=20,  # Need logprobs for uncertainty estimation
        )
        
        # Create WhiteboxModelvLLM for lm_polygraph
        self.model = WhiteboxModelvLLM(
            self.llm, 
            self.sampling_params, 
            device=self.device
        )
        
        tqdm.write(f"vLLM model loaded: {self.model_name}")

    def prepare_chat_messages(self, question: str, doc_text: str = None) -> List[Dict[str, str]]:
        """Prepare chat messages with or without document context."""
        if doc_text:
            content = f"""Answer the following question based on the given document. Only give me the answer and do not output any other words.

The following are given documents: {doc_text}

Question: {question}"""
        else:
            content = f"""Answer the following question. Only give me the answer and do not output any other words.

Question: {question}"""
            
        # Format as chat messages
        messages = [{"role": "user", "content": content}]
        return messages

    def generate(
        self,
        question: str,
        doc_text: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """Generate a response using the vLLM model."""
        if temperature is not None:
            # Create new sampling params with different temperature
            sampling_params = SamplingParams(
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
                logprobs=20
            )
        else:
            sampling_params = self.sampling_params
        
        # Prepare messages
        messages = self.prepare_chat_messages(question, doc_text)
        
        # Convert to prompt format
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Generate with vLLM
        outputs = self.llm.generate([prompt], sampling_params)
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text
        
        # For compatibility, also try to get tokens if needed
        try:
            generated_tokens = self.tokenizer.encode(generated_text)
        except:
            generated_tokens = []
        
        return {
            "text": generated_text,
            "tokens": generated_tokens
        }


class PolygraphCalculator:
    """Handles uncertainty estimation using lm_polygraph library with vLLM."""

    def __init__(self, vllm_model: VLLMModel, batch_size: int = 4):
        self.vllm_model = vllm_model
        self.batch_size = batch_size
        
        # Set up NLI model for semantic metrics
        self.nli_model = Deberta(device='cuda:1', batch_size=5)
        self.nli_model.setup()
        
        self.initialize_estimators()
    
    def initialize_estimators(self):
        """Initialize all uncertainty estimators from lm_polygraph."""
        # Initialize stat calculators for vLLM
        self.calc_infer_llm = GreedyProbsCalculator()
        self.calc_nli = GreedyAlternativesNLICalculator(nli_model=self.nli_model, batch_size=300)
        self.calc_samples = SamplingGenerationCalculator()
        self.calc_cross_encoder = CrossEncoderSimilarityMatrixCalculator(device='cuda:3', batch_size=100)
        self.calc_semantic_matrix = SemanticMatrixCalculator(nli_model=self.nli_model, batch_size=2)
        self.calc_semantic_classes = SemanticClassesCalculator()
        self.calc_entropy = EntropyCalculator()
        self.calc_prompt = PromptCalculator()
        
        # Initialize estimators
        self.estimators = [
            MaximumSequenceProbability(),
            Perplexity(),
            MeanTokenEntropy(),
            MonteCarloSequenceEntropy(), 
            MonteCarloNormalizedSequenceEntropy(),
            RenyiNeg(),
            FisherRao(),
            SemanticEntropy(),
            ClaimConditionedProbability(),
            TokenSAR(),
            SentenceSAR(),
            SAR(),
            EigenScore(),
            PTrue(),
            NumSemSets(),
            EigValLaplacian(),
            DegMat(),
            Eccentricity(),
            LexicalSimilarity(),
            KernelLanguageEntropy(),
            LUQ()
        ]
    
    def calculate_uncertainty_batch(self, prompts: List[str]) -> List[Dict[str, float]]:
        """Calculate uncertainty metrics for a batch of prompts using vLLM."""
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Processing scenarios in texts", total=len(prompts) // self.batch_size, leave=False):
            batch = prompts[i:i+self.batch_size]
            
            # Initialize dependencies with input texts
            deps = {"input_texts": batch}
            
            # 测量每个计算器的执行时间
            calculators = [
                ("calc_infer_llm", lambda: self.calc_infer_llm(deps, texts=batch, model=self.vllm_model.model)),
                ("calc_nli", lambda: self.calc_nli(deps, texts=batch, model=self.vllm_model.model)),
                ("calc_samples", lambda: self.calc_samples(deps, texts=batch, model=self.vllm_model.model)),
                ("calc_cross_encoder", lambda: self.calc_cross_encoder(deps, texts=batch, model=self.vllm_model.model)),
                ("calc_semantic_matrix", lambda: self.calc_semantic_matrix(deps, texts=batch, model=self.vllm_model.model)),
                ("calc_semantic_classes", lambda: self.calc_semantic_classes(deps, texts=batch, model=self.vllm_model.model)),
                ("calc_entropy", lambda: self.calc_entropy(deps, texts=batch, model=self.vllm_model.model)),
                ("calc_prompt", lambda: self.calc_prompt(deps, texts=batch, model=self.vllm_model.model)),
            ]
            
            for name, calculator in tqdm(calculators):
                tqdm.write(f"Calculating {name}...")
                start_time = time.time()
                update = calculator()
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"{name} 执行时间: {execution_time:.4f} 秒")
                deps.update(update)
            
            # Calculate uncertainty scores for each estimator
            batch_results = []
            for _ in batch:
                batch_results.append({})
            
            # Apply each estimator
            for estimator in tqdm(self.estimators):
                try:
                    uncertainty_scores = estimator(deps)
                    estimator_name = str(estimator)
                    
                    # Add to results
                    for j, score in enumerate(uncertainty_scores):
                        if isinstance(score, dict):
                            for key, value in score.items():
                                batch_results[j][f"{estimator_name}_{key}"] = value
                        else:
                            batch_results[j][estimator_name] = score
                except Exception as e:
                    tqdm.write(f"Error in estimator {estimator}: {e}")
            
            results.extend(batch_results)
        
        return results
    
    def calculate_uncertainty_all_scenarios(
        self,
        question: str,
        doc_text: str,
        individual_doc_texts: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Calculate uncertainty for all three scenarios in one batch:
        1. No document (naive)
        2. All documents (rag)
        3. Each individual document
        
        Returns a tuple of (naive_metrics, rag_metrics, individual_docs_metrics)
        """
        # Prepare chat messages for all scenarios
        messages = []
        
        # Naive scenario (no documents)
        naive_messages = self.vllm_model.prepare_chat_messages(question)
        naive_prompt = self.vllm_model.tokenizer.apply_chat_template(naive_messages, tokenize=False)
        messages.append(naive_prompt)
        
        # RAG scenario (all documents)
        rag_messages = self.vllm_model.prepare_chat_messages(question, doc_text)
        rag_prompt = self.vllm_model.tokenizer.apply_chat_template(rag_messages, tokenize=False)
        messages.append(rag_prompt)
        
        # Individual document scenarios
        for individual_doc_text in individual_doc_texts:
            individual_messages = self.vllm_model.prepare_chat_messages(question, individual_doc_text)
            individual_prompt = self.vllm_model.tokenizer.apply_chat_template(individual_messages, tokenize=False)
            messages.append(individual_prompt)
        
        # Calculate uncertainty for all prompts in one batch
        all_uncertainties = self.calculate_uncertainty_batch(messages)
        
        # Extract results by scenario
        naive_metrics = all_uncertainties[0]
        rag_metrics = all_uncertainties[1]
        individual_docs_metrics = all_uncertainties[2:] if len(all_uncertainties) > 2 else []
        
        return naive_metrics, rag_metrics, individual_docs_metrics


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


class RAGEvaluator:
    """Main class for evaluating RAG performance using lm_polygraph with vLLM."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        model_device: str = "cuda:0",
        gpu_memory_utilization: float = 0.7,
        batch_size: int = 4
    ):
        self.doc_processor = DocumentProcessor()
        self.batch_size = batch_size
        
        # Initialize the vLLM model
        self.vllm_model = VLLMModel(
            model_name=model_name,
            device=model_device,
            max_tokens=max_tokens,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization
        )

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize Polygraph calculator
        tqdm.write("Initializing lm_polygraph calculator with vLLM...")
        self.polygraph_calculator = PolygraphCalculator(
            self.vllm_model, 
            batch_size=self.batch_size
        )

    def _get_response(self, question: str, doc_text: str = None) -> Dict[str, Any]:
        """Get a response from the vLLM model."""
        # Generate response
        generation = self.vllm_model.generate(
            question=question,
            doc_text=doc_text,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            "response": generation["text"],
        }

    def evaluate_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with its retrieved documents using batched processing."""
        question = item['question']
        golden_answers = item['golden_answers']
        retrieved_docs = item['output']['retrieval_result']

        # Format document texts for all scenarios
        all_docs_text = self.doc_processor.format_documents(retrieved_docs)
        
        individual_doc_texts = []
        for i, doc in enumerate(retrieved_docs):
            # Use the original doc index (i+1)
            single_doc_text = self.doc_processor.format_documents([doc], start_index=i+1)
            individual_doc_texts.append(single_doc_text)
        
        # Calculate all uncertainty metrics in one batch
        naive_metrics, rag_metrics, individual_docs_metrics = self.polygraph_calculator.calculate_uncertainty_all_scenarios(
            question=question,
            doc_text=all_docs_text,
            individual_doc_texts=individual_doc_texts
        )
        
        # Generate responses
        naive_response = self._get_response(question)["response"]
        rag_response = self._get_response(question, all_docs_text)["response"]
        
        # 计算效用指标 (reductions)
        utility = {}
        for metric_name in naive_metrics:
            if metric_name in rag_metrics:
                reduction_name = f"{metric_name}_reduction"
                
                # 对于类似熵的指标，naive - rag 是减少 (正值表示减少不确定性)
                # 对于信心类指标，rag - naive 是增益 (正值表示增加信心)
                if any(entropy_term in metric_name.lower() for entropy_term in ["entropy", "perplexity", "uncertainty"]):
                    utility[reduction_name] = float(naive_metrics[metric_name]) - float(rag_metrics[metric_name])
                else:
                    utility[reduction_name] = float(rag_metrics[metric_name]) - float(naive_metrics[metric_name])
        
        # 处理单个文档结果
        individual_doc_results = []
        for i, doc in enumerate(retrieved_docs):
            if i < len(individual_docs_metrics):
                doc_metrics = individual_docs_metrics[i]
                
                # 为单个文档生成响应
                doc_response = self._get_response(question, individual_doc_texts[i])["response"]
                
                # 计算单个文档的效用
                doc_utility = {}
                for metric_name in naive_metrics:
                    if metric_name in doc_metrics:
                        reduction_name = f"{metric_name}_reduction"
                        
                        # 同样的逻辑
                        if any(entropy_term in metric_name.lower() for entropy_term in ["entropy", "perplexity", "uncertainty"]):
                            doc_utility[reduction_name] = float(naive_metrics[metric_name]) - float(doc_metrics[metric_name])
                        else:
                            doc_utility[reduction_name] = float(doc_metrics[metric_name]) - float(naive_metrics[metric_name])
                
                # 包含原始数据中的检索器分数
                doc_utility["retriever_score"] = float(doc.get('score', 0.0))
                
                doc_result = {
                    "doc_index": i,
                    "doc_id": doc.get('id', f"doc_{i}"),
                    "doc_score": float(doc.get('score', 0.0)),
                    "doc_content": doc.get('contents', ''),
                    "response": doc_response,
                    "metrics": doc_metrics,
                    "utility": doc_utility
                }
                
                individual_doc_results.append(doc_result)
        
        # 构建最终结果
        result = {
            "id": item['id'],
            "question": question,
            "golden_answers": golden_answers,
            "naive_response": naive_response,
            "rag_response": rag_response,
            "naive_metrics": naive_metrics,
            "rag_metrics": rag_metrics,
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
        # Load dataset
        data = load_dataset(dataset_path)
        if max_samples:
            data = data[:max_samples]
        
        # Get output directory and base filename
        output_dir = os.path.dirname(output_path)
        base_output_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Check for existing temporary files to determine starting point
        existing_results = []
        start_index = 0
        
        if os.path.exists(output_dir):
            # Find all temp files in the directory
            temp_files = [f for f in os.listdir(output_dir) if f.startswith(f"{base_output_name}_temp_") and f.endswith(".json")]
            
            if temp_files:
                # Extract indices from filenames
                indices = []
                for temp_file in temp_files:
                    try:
                        # Extract index number from filename (e.g., _temp_42.json -> 42)
                        idx_str = temp_file.replace(f"{base_output_name}_temp_", "").replace(".json", "")
                        indices.append(int(idx_str))
                    except ValueError:
                        continue
                
                if indices:
                    # Find the highest index
                    latest_index = max(indices)
                    tqdm.write(f"Found existing temporary files. Resuming from item {latest_index + 1}")
                    
                    # Load the latest temporary file to get existing results
                    latest_temp_file = f"{base_output_name}_temp_{latest_index}.json"
                    latest_temp_path = os.path.join(output_dir, latest_temp_file)
                    
                    try:
                        # Use custom JSON loader to handle Infinity values
                        existing_results = load_dataset(latest_temp_path)
                        
                        # Set the starting index for the loop
                        start_index = latest_index
                    except Exception as e:
                        tqdm.write(f"Error loading existing results: {e}")
                        tqdm.write("Starting from the beginning")
                        start_index = 0
                        existing_results = []
        else:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
        
        results = existing_results

        # Process from the next unprocessed item
        for i in range(start_index, len(data)):
            item = data[i]
            tqdm.write(f"Processing question {i+1}/{len(data)}: {item['question'][:50]}...")
            
            # Process single item
            result = self.evaluate_question(item)
            results.append(result)
            
            # Save intermediate results after each question
            temp_output_path = os.path.join(output_dir, f"{base_output_name}_temp_{i+1}.json")
            tqdm.write(f"Saving intermediate results to {temp_output_path}")
            
            # Use the custom JSON saver to handle Infinity values
            save_json_with_infinity(results, temp_output_path)

        # Save final results with custom JSON saver
        save_json_with_infinity(results, output_path)

        tqdm.write(f"Evaluation complete, results saved to {output_path}")


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG performance with vLLM")
    parser.add_argument("--dataset_path", type=str,
                        default="data/nq_2024_11_07_12_14_naive/intermediate_data.json")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_path", type=str,
                        default="data/nq_2024_11_07_12_14_naive/results/rag_evaluation_results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model_device", type=str, default="cuda:4",
                        help="Device for model calculations")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM (0.0 to 1.0)")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for processing scenarios in one question")

    args = parser.parse_args()

    evaluator = RAGEvaluator(
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        model_device=args.model_device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size
    )

    evaluator.evaluate_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()