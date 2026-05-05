import os
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
from openai import OpenAI


def _embed_query(query: str) -> List[float]:
    """Embed a query string using the OpenAI SDK.

    The OpenAI SDK automatically picks up OPENAI_API_KEY and OPENAI_BASE_URL
    from the environment, so this transparently uses the Vocareum proxy when
    OPENAI_BASE_URL is set.
    """
    client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from env
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=query)
    return resp.data[0].embedding


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends.

    Searches both the current working directory and the directory where this
    module lives, so the app works regardless of where Streamlit is launched
    from (e.g. `streamlit run project/chat.py` vs `cd project && streamlit run chat.py`).
    """
    backends: Dict[str, Dict[str, str]] = {}

    search_dirs = []
    seen = set()
    for d in (Path("."), Path(__file__).resolve().parent):
        d_resolved = d.resolve()
        if d_resolved not in seen:
            seen.add(d_resolved)
            search_dirs.append(d_resolved)

    candidate_dirs = []
    for base in search_dirs:
        if not base.exists():
            continue
        for d in base.iterdir():
            if d.is_dir() and "chroma" in d.name.lower():
                candidate_dirs.append(d)

    for chroma_dir in candidate_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collections = client.list_collections()

            for collection in collections:
                # Collection objects in newer chromadb expose `.name`; fall back to str
                collection_name = getattr(collection, "name", str(collection))
                key = f"{chroma_dir.name}::{collection_name}"

                try:
                    count = client.get_collection(collection_name).count()
                except Exception:
                    count = "?"

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection_name,
                    "display_name": f"{chroma_dir.name} / {collection_name} ({count} docs)",
                    "document_count": str(count),
                }
        except Exception as e:
            # Fallback entry for inaccessible directories
            err_msg = str(e)
            if len(err_msg) > 60:
                err_msg = err_msg[:57] + "..."
            backends[f"{chroma_dir.name}::error"] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (error: {err_msg})",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend.

    Returns a tuple of (collection, success, error_message).
    """
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    where_filter = None
    if mission_filter and mission_filter.lower() not in ("all", "none", ""):
        where_filter = {"mission": mission_filter}

    # Embed the query ourselves so we always use OPENAI_BASE_URL (Vocareum)
    # rather than whatever embedding function may be baked into the collection.
    query_embedding = _embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )
    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts: List[str] = ["# Retrieved NASA Mission Context\n"]

    for idx, (doc, meta) in enumerate(zip(documents, metadatas or []), start=1):
        meta = meta or {}
        mission = str(meta.get("mission", "unknown")).replace("_", " ").title()
        source = meta.get("source", "unknown")
        category = str(meta.get("document_category", "general")).replace("_", " ").title()

        header = f"## Source [{idx}] — Mission: {mission} | File: {source} | Category: {category}"
        context_parts.append(header)

        # Truncate very long documents to keep prompt sizes reasonable
        max_chars = 1500
        body = doc if len(doc) <= max_chars else doc[:max_chars] + " ...[truncated]"
        context_parts.append(body)
        context_parts.append("")  # blank line separator

    return "\n".join(context_parts)
