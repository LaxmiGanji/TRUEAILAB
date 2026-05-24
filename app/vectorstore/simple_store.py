import os
import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    def __init__(self, db_path: str = "vectorstore.db"):
        self.db_path = db_path
        # Ensure parent directory exists
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def clear(self):
        """Clears all records in the vector store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks")
            conn.commit()
        logger.info("Cleared all chunks from the vector database.")

    def add_chunk(self, doc_title: str, content: str, embedding: List[float], metadata: Dict[str, Any]):
        """Saves a chunk and its embedding to the SQLite database."""
        # Convert embedding to numpy float32 bytes for storage efficiency
        emb_arr = np.array(embedding, dtype=np.float32)
        emb_bytes = emb_arr.tobytes()
        metadata_str = json.dumps(metadata)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chunks (doc_title, content, embedding, metadata_json) VALUES (?, ?, ?, ?)",
                (doc_title, content, emb_bytes, metadata_str)
            )
            conn.commit()

    def get_chunk_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]

    def similarity_search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs Cosine Similarity search against all stored vectors.
        Returns a list of dictionaries with matching chunk info and similarity scores.
        """
        query_vec = np.array(query_embedding, dtype=np.float32)
        norm_query = np.linalg.norm(query_vec)
        
        if norm_query == 0:
            logger.warning("Query vector norm is zero. Similarity search may fail.")
            return []

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_title, content, embedding, metadata_json FROM chunks")
            rows = cursor.fetchall()

            for doc_title, content, emb_bytes, metadata_json in rows:
                # Reconstruct embedding from bytes
                emb_vec = np.frombuffer(emb_bytes, dtype=np.float32)
                norm_emb = np.linalg.norm(emb_vec)
                
                if norm_emb == 0:
                    similarity = 0.0
                else:
                    dot_product = np.dot(query_vec, emb_vec)
                    similarity = float(dot_product / (norm_query * norm_emb))

                try:
                    metadata = json.loads(metadata_json)
                except Exception:
                    metadata = {}

                results.append({
                    "title": doc_title,
                    "content": content,
                    "score": similarity,
                    "metadata": metadata
                })

        # Sort by similarity score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Log all scores for visibility
        logger.info("Similarity Search Results:")
        for idx, res in enumerate(results[:top_k]):
            logger.info(f"Top {idx+1}: Score: {res['score']:.4f} | Title: {res['title']} | Snippet: {res['content'][:60]}...")
            
        return results[:top_k]
