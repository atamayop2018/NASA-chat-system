import math
import os
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load variables from a local .env file if one is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# RAGAS imports
try:
    from ragas import SingleTurnSample, evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        ResponseRelevancy,
        Faithfulness,
        BleuScore,
        RougeScore,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    try:
        # Resolve credentials/models from environment so callers don't have to
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OPENAI_API_KEY not set"}

        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

        chat_kwargs = {"model": chat_model, "temperature": 0, "api_key": api_key}
        embed_kwargs = {"model": embedding_model, "api_key": api_key}
        if base_url:
            chat_kwargs["base_url"] = base_url
            embed_kwargs["base_url"] = base_url

        # Create evaluator LLM and embeddings
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(**chat_kwargs))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(**embed_kwargs))

        # Build a single-turn sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts or [],
            reference="\n".join(contexts) if contexts else answer,
        )

        # Define metrics that need an LLM and/or embeddings
        relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        faithfulness = Faithfulness(llm=evaluator_llm)
        bleu = BleuScore()
        rouge = RougeScore()

        dataset = EvaluationDataset(samples=[sample])

        result = evaluate(
            dataset=dataset,
            metrics=[relevancy, faithfulness, bleu, rouge],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        # Extract scores into a flat dict
        scores: Dict[str, float] = {}
        try:
            df = result.to_pandas()
            for col in df.columns:
                if col in ("user_input", "response", "retrieved_contexts", "reference"):
                    continue
                val = df[col].iloc[0]
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                # Drop NaN / inf so the UI doesn't blow up on the progress bar
                if math.isnan(fval) or math.isinf(fval):
                    continue
                scores[col] = fval
        except Exception:
            # Fallback: try dict-like access
            try:
                for k, v in dict(result).items():
                    try:
                        scores[k] = float(v)
                    except (TypeError, ValueError):
                        continue
            except Exception:
                pass

        return scores if scores else {"error": "No scores produced"}

    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
