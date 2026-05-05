#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""

    def __init__(self,
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize the embedding pipeline."""
        # OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Configuration
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_persist_directory = chroma_persist_directory

        # ChromaDB client (persistent)
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "NASA space mission text documents"},
        )

        logger.info(
            f"Pipeline initialized: collection='{collection_name}', "
            f"persist_dir='{chroma_persist_directory}', model='{embedding_model}'"
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Split text into chunks with metadata.

        Guarantees:
          1. No produced chunk exceeds ``self.chunk_size`` characters.
          2. Consecutive chunks share exactly ``self.chunk_overlap`` characters
             byte-for-byte (i.e. ``chunks[i][-overlap:] == chunks[i+1][:overlap]``).
          3. Where possible, chunks end at a sentence boundary (``. ``, ``! ``,
             ``? ``, ``\\n\\n``, ``\\n``) within the size cap; this only ever
             *shrinks* a chunk, never grows it, so invariant (1) is preserved.

        Note: text is intentionally NOT ``.strip()``-ed and chunks are NOT
        stripped, so that the byte-exact overlap invariant holds.
        """
        text = text or ""
        if not text:
            return []

        # Validate config
        chunk_size = max(1, int(self.chunk_size))
        chunk_overlap = max(0, int(self.chunk_overlap))
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 2  # safety: overlap must be < size

        # Short texts: no chunking needed
        if len(text) <= chunk_size:
            return [(text, {**metadata, "chunk_index": 0, "chunk_count": 1})]

        chunks: List[Tuple[str, Dict[str, Any]]] = []
        start = 0
        chunk_index = 0
        text_len = len(text)

        while start < text_len:
            # Hard cap: end can never exceed start + chunk_size
            end = min(start + chunk_size, text_len)

            # Try to nudge `end` back to a sentence boundary inside the cap.
            # rfind returns positions where the terminator fully fits before `end`,
            # so the resulting break is always <= end (chunk stays within the cap).
            if end < text_len:
                window_start = max(start + max(1, chunk_size // 2), end - 200)
                best_break = -1
                for terminator in (". ", "! ", "? ", "\n\n", "\n"):
                    idx = text.rfind(terminator, window_start, end)
                    if idx > best_break:
                        best_break = idx + len(terminator)
                if best_break > start:
                    end = best_break

            chunk = text[start:end]
            # Defensive assertion of invariant (1)
            assert len(chunk) <= chunk_size, (
                f"Chunk length {len(chunk)} exceeds chunk_size {chunk_size}"
            )
            chunks.append((chunk, {**metadata, "chunk_index": chunk_index}))
            chunk_index += 1

            if end >= text_len:
                break

            # Advance: next chunk starts `chunk_overlap` chars before current end.
            # This guarantees byte-exact overlap (invariant 2).
            next_start = end - chunk_overlap
            # Guard against pathological input where sentence-boundary nudging
            # would push us backwards or stall the loop.
            if next_start <= start:
                next_start = start + max(1, chunk_size - chunk_overlap)
            start = next_start

        total = len(chunks)
        for _, meta in chunks:
            meta["chunk_count"] = total

        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """Check if a document with the given ID already exists in the collection."""
        try:
            result = self.collection.get(ids=[doc_id])
            return bool(result and result.get("ids"))
        except Exception as e:
            logger.debug(f"Existence check failed for {doc_id}: {e}")
            return False

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing document in the collection."""
        try:
            embedding = self.get_embedding(text)
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """Delete all documents from a specific source."""
        try:
            all_docs = self.collection.get()
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(
                    f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}"
                )
                return len(ids_to_delete)
            logger.info(f"No documents found matching source pattern: {source_pattern}")
            return 0
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """Get all document IDs for a specific file."""
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            all_docs = self.collection.get()
            file_doc_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == source and
                        metadata.get('mission') == mission):
                    file_doc_ids.append(all_docs['ids'][i])
            return file_doc_ids
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """Generate stable document ID based on file path and chunk position."""
        mission = metadata.get("mission", self.extract_mission_from_path(file_path))
        source = metadata.get("source", file_path.stem)
        chunk_index = int(metadata.get("chunk_index", 0))
        return f"{mission}_{source}_chunk_{chunk_index:04d}"

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """Process plain text files with enhanced metadata extraction."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return []

            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        return 'unknown'

    def extract_data_type_from_path(self, file_path: Path) -> str:
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        return 'document'

    def extract_document_category_from_filename(self, filename: str) -> str:
        filename_lower = filename.lower()
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        elif 'full_text' in filename_lower:
            return 'complete_document'
        return 'general_document'

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """Scan data directories for text files only."""
        base_path = Path(base_path)
        files_to_process: List[Path] = []
        data_dirs = ['apollo11', 'apollo13', 'challenger']

        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if not dir_path.exists():
                # also try data_text/ subdir
                alt = base_path / 'data_text' / data_dir
                if alt.exists():
                    dir_path = alt
                else:
                    continue
            logger.info(f"Scanning directory: {dir_path}")
            text_files = list(dir_path.glob('**/*.txt'))
            files_to_process.extend(text_files)
            logger.info(f"Found {len(text_files)} text files in {data_dir}")

        filtered_files = []
        for file_path in files_to_process:
            if (file_path.name.startswith('.') or
                    'summary' in file_path.name.lower() or
                    file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)

        logger.info(f"Total text files to process: {len(filtered_files)}")
        return filtered_files

    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]],
                                    file_path: Path, batch_size: int = 50,
                                    update_mode: str = 'skip') -> Dict[str, int]:
        """Add documents to ChromaDB collection in batches with update handling."""
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}

        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        # 'replace' mode: delete all existing chunks for this file first
        if update_mode == 'replace':
            existing_ids = self.get_file_documents(file_path)
            if existing_ids:
                try:
                    self.collection.delete(ids=existing_ids)
                    logger.info(f"Replaced {len(existing_ids)} existing chunks for {file_path.name}")
                except Exception as e:
                    logger.error(f"Error deleting existing docs for replace: {e}")

        # Process in batches
        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start:batch_start + batch_size]
            batch_ids: List[str] = []
            batch_texts: List[str] = []
            batch_metas: List[Dict[str, Any]] = []
            batch_embeddings: List[List[float]] = []

            for text, metadata in batch:
                doc_id = self.generate_document_id(file_path, metadata)
                exists = self.check_document_exists(doc_id) if update_mode != 'replace' else False

                if exists and update_mode == 'skip':
                    stats['skipped'] += 1
                    continue

                try:
                    embedding = self.get_embedding(text)
                except Exception:
                    continue

                if exists and update_mode == 'update':
                    if self.update_document(doc_id, text, metadata):
                        stats['updated'] += 1
                    continue

                # Queue for batch insert
                batch_ids.append(doc_id)
                batch_texts.append(text)
                batch_metas.append(metadata)
                batch_embeddings.append(embedding)

            if batch_ids:
                try:
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_texts,
                        metadatas=batch_metas,
                        embeddings=batch_embeddings,
                    )
                    stats['added'] += len(batch_ids)
                except Exception as e:
                    logger.error(f"Error adding batch: {e}")

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = 'skip') -> Dict[str, Any]:
        """Process all text files and add to ChromaDB."""
        stats: Dict[str, Any] = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        files = self.scan_text_files_only(base_path)
        for file_path in files:
            try:
                chunks = self.process_text_file(file_path)
                if not chunks:
                    continue
                file_stats = self.add_documents_to_collection(
                    chunks, file_path, update_mode=update_mode
                )
                stats['files_processed'] += 1
                stats['total_chunks'] += len(chunks)
                stats['documents_added'] += file_stats['added']
                stats['documents_updated'] += file_stats['updated']
                stats['documents_skipped'] += file_stats['skipped']

                mission = self.extract_mission_from_path(file_path)
                m = stats['missions'].setdefault(
                    mission, {'files': 0, 'chunks': 0, 'added': 0, 'updated': 0, 'skipped': 0}
                )
                m['files'] += 1
                m['chunks'] += len(chunks)
                m['added'] += file_stats['added']
                m['updated'] += file_stats['updated']
                m['skipped'] += file_stats['skipped']
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats['errors'] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""
        try:
            return {
                'collection_name': self.collection_name,
                'document_count': self.collection.count(),
                'persist_directory': self.chroma_persist_directory,
            }
        except Exception as e:
            return {'error': str(e)}

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the collection for testing."""
        try:
            embedding = self.get_embedding(query_text)
            return self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
            )
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            all_docs = self.collection.get()
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}

            stats: Dict[str, Any] = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {},
            }

            for metadata in all_docs['metadatas']:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')

                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1

            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='./data_text', help='Path to data directories')
    parser.add_argument('--openai-key', required=True, help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                        help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')

    args = parser.parse_args()

    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return

    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return

    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)
    processing_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")

    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")

    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):
                logger.info(f"Result {i+1}: {doc[:200]}...")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
