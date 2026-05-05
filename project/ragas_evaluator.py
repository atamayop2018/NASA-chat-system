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


def batch_evaluate_responses(samples: List[Dict[str, object]]) -> Dict[str, object]:
    """Run RAGAS metrics over a batch of samples in a single ``evaluate()`` call.

    Args:
        samples: list of dicts with keys ``user_input``, ``response``,
                 ``retrieved_contexts`` (list[str]).

    Returns:
        dict with:
          - ``per_sample``: list of dicts (one per input sample) keyed by metric name
          - ``aggregate``:  dict of {metric_name: mean_score}
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    if not samples:
        return {"error": "No samples provided"}

    try:
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )

        ragas_samples = [
            SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=list(s.get("retrieved_contexts") or []),
            )
            for s in samples
        ]
        dataset = EvaluationDataset(samples=ragas_samples)

        metrics = [
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
        ]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            show_progress=False,
        )

        df = result.to_pandas()
        skip_cols = {"user_input", "response", "retrieved_contexts", "reference"}
        metric_cols = [c for c in df.columns if c not in skip_cols]

        per_sample: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            row_scores: Dict[str, float] = {}
            for c in metric_cols:
                try:
                    row_scores[c] = float(row[c])
                except Exception:
                    pass
            per_sample.append(row_scores)

        aggregate: Dict[str, float] = {}
        for c in metric_cols:
            try:
                aggregate[c] = float(df[c].mean())
            except Exception:
                pass

        return {"per_sample": per_sample, "aggregate": aggregate}

    except Exception as e:
        return {"error": f"Batch evaluation failed: {str(e)}"}
