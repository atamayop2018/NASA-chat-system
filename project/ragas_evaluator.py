from typing import Dict, List

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS imports
try:
    from ragas import SingleTurnSample, evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import (
        ResponseRelevancy,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics."""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not contexts:
        return {"error": "No retrieved contexts available for evaluation"}

    try:
        # Wrap the evaluator LLM and embeddings for RAGAS
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )

        # Build a single-turn evaluation sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=list(contexts),
        )
        dataset = EvaluationDataset(samples=[sample])

        # Define metric instances
        metrics = [
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
        ]

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            show_progress=False,
        )

        # Convert to a flat dict of floats
        try:
            df = result.to_pandas()
            scores: Dict[str, float] = {}
            for col in df.columns:
                if col in ("user_input", "response", "retrieved_contexts", "reference"):
                    continue
                try:
                    scores[col] = float(df[col].iloc[0])
                except Exception:
                    continue
            return scores or {"error": "No numeric metrics returned"}
        except Exception:
            # Fallback: try dict-like access
            return {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}

    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
