import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")

    # Look for ChromaDB directories matching common naming patterns
    candidate_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and (
            d.name.startswith("chroma_db") or d.name.startswith("chromadb")
        )
    ]

    # Loop through each discovered directory
    for chroma_dir in candidate_dirs:
        try:
            # Initialize a ChromaDB client pointing at this directory
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            # Retrieve list of available collections
            collections = client.list_collections()

            # Loop through each collection
            for collection in collections:
                # Some chromadb versions return objects, others return strings
                col_name = getattr(collection, "name", collection)

                # Unique key combining directory and collection name
                backend_key = f"{chroma_dir.name}::{col_name}"

                # Try to get the document count
                try:
                    col_obj = client.get_collection(name=col_name)
                    doc_count = col_obj.count()
                except Exception:
                    doc_count = "unknown"

                backends[backend_key] = {
                    "directory": str(chroma_dir),
                    "collection_name": col_name,
                    "display_name": f"{chroma_dir.name} / {col_name} ({doc_count} docs)",
                    "document_count": str(doc_count),
                }

        except Exception as e:
            # Fallback entry for inaccessible directories
            err = str(e)
            if len(err) > 80:
                err = err[:77] + "..."
            backends[f"{chroma_dir.name}::error"] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (error: {err})",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[object, bool, str]:
    """Initialize the RAG system with the specified backend.

    Returns a tuple of (collection, success, error_message).
    """
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name)
        return collection, True, ""
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter to None (no filtering)
    where_filter = None

    # Apply a mission filter if one was provided and isn't a wildcard
    if mission_filter and mission_filter.lower() not in ("all", "any", ""):
        where_filter = {"mission": mission_filter}

    # Execute the query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts: List[str] = ["Retrieved context from NASA archives:"]

    for i, (doc, meta) in enumerate(zip(documents, metadatas or [])):
        meta = meta or {}

        # Mission
        mission = meta.get("mission", "unknown")
        mission_display = mission.replace("_", " ").title()

        # Source
        source = meta.get("source", "unknown source")

        # Category
        category = meta.get("document_category", "general")
        category_display = category.replace("_", " ").title()

        header = f"\n[Source {i + 1}] Mission: {mission_display} | Document: {source} | Category: {category_display}"
        context_parts.append(header)

        # Truncate very long chunks to keep context manageable
        max_len = 1500
        if doc and len(doc) > max_len:
            context_parts.append(doc[:max_len] + "...")
        else:
            context_parts.append(doc or "")

    return "\n".join(context_parts)
