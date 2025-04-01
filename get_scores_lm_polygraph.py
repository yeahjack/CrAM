import argparse
import json
import os
import torch
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

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
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator


class VLLMModel:
    """Handles direct interaction with vLLM models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        max_tokens: int = 512,
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
            logprobs=20  # Need logprobs for uncertainty estimation
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
        self.nli_model = Deberta(device='cuda:1')
        self.nli_model.setup()
        
        self.initialize_estimators()
    
    def initialize_estimators(self):
        """Initialize all uncertainty estimators from lm_polygraph."""
        # Initialize stat calculators for vLLM
        self.calc_infer_llm = GreedyProbsCalculator()
        self.calc_nli = GreedyAlternativesNLICalculator(nli_model=self.nli_model)
        self.calc_samples = SamplingGenerationCalculator()
        self.calc_cross_encoder = CrossEncoderSimilarityMatrixCalculator()
        self.calc_semantic_matrix = SemanticMatrixCalculator(nli_model=self.nli_model)
        self.calc_semantic_classes = SemanticClassesCalculator()
        
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
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            
            # Initialize dependencies with input texts
            deps = {"input_texts": batch}
            
            # Run through all calculators
            deps.update(self.calc_infer_llm(deps, texts=batch, model=self.vllm_model.model))
            deps.update(self.calc_nli(deps, texts=batch, model=self.vllm_model.model))
            deps.update(self.calc_samples(deps, texts=batch, model=self.vllm_model.model))
            deps.update(self.calc_cross_encoder(deps, texts=batch, model=self.vllm_model.model))
            deps.update(self.calc_semantic_matrix(deps, texts=batch, model=self.vllm_model.model))
            deps.update(self.calc_semantic_classes(deps, texts=batch, model=self.vllm_model.model))
            
            # Calculate uncertainty scores for each estimator
            batch_results = []
            for _ in batch:
                batch_results.append({})
            
            # Apply each estimator
            for estimator in self.estimators:
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
        
        # Calculate utility metrics (reductions)
        utility = {}
        for metric_name in naive_metrics:
            if metric_name in rag_metrics:
                reduction_name = f"{metric_name}_reduction"
                
                # For entropy-like metrics, naive - rag is the reduction (positive means reduced uncertainty)
                # For confidence-like metrics, rag - naive is the gain (positive means increased confidence)
                if any(entropy_term in metric_name.lower() for entropy_term in ["entropy", "perplexity", "uncertainty"]):
                    utility[reduction_name] = naive_metrics[metric_name] - rag_metrics[metric_name]
                else:
                    utility[reduction_name] = rag_metrics[metric_name] - naive_metrics[metric_name]
        
        # Process individual document results
        individual_doc_results = []
        for i, doc in enumerate(retrieved_docs):
            if i < len(individual_docs_metrics):
                doc_metrics = individual_docs_metrics[i]
                
                # Generate response for individual document
                doc_response = self._get_response(question, individual_doc_texts[i])["response"]
                
                # Calculate utility for individual document
                doc_utility = {}
                for metric_name in naive_metrics:
                    if metric_name in doc_metrics:
                        reduction_name = f"{metric_name}_reduction"
                        
                        # Same logic as above
                        if any(entropy_term in metric_name.lower() for entropy_term in ["entropy", "perplexity", "uncertainty"]):
                            doc_utility[reduction_name] = naive_metrics[metric_name] - doc_metrics[metric_name]
                        else:
                            doc_utility[reduction_name] = doc_metrics[metric_name] - naive_metrics[metric_name]
                
                # Include retriever score from original data
                doc_utility["retriever_score"] = doc.get('score', 0.0)
                
                doc_result = {
                    "doc_index": i,
                    "doc_id": doc.get('id', f"doc_{i}"),
                    "doc_score": doc.get('score', 0.0),
                    "doc_content": doc.get('contents', ''),
                    "response": doc_response,
                    "metrics": doc_metrics,
                    "utility": doc_utility
                }
                
                individual_doc_results.append(doc_result)
        
        # Build final result
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

        results = []

        # Process one by one
        for i, item in enumerate(tqdm(data, desc="Processing questions")):
            tqdm.write(f"Processing question {i+1}/{len(data)}: {item['question'][:50]}...")
            
            # Process single item
            result = self.evaluate_question(item)
            results.append(result)
            
            # Save intermediate results after each question
            temp_output_path = f"{os.path.splitext(output_path)[0]}_temp_{i+1}.json"
            tqdm.write(f"Saving intermediate results to {temp_output_path}")
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        # Save final results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

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
    parser.add_argument("--model_device", type=str, default="cuda:0",
                        help="Device for model calculations")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                        help="GPU memory utilization for vLLM (0.0 to 1.0)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing multiple examples")

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