import os
import json
import logging
from typing import List, Dict, Any, Tuple
from app.vectorstore.simple_store import SimpleVectorStore
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.utils.session import SessionManager
from app.prompts.templates import RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(
        self,
        vector_store: SimpleVectorStore,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        session_manager: SessionManager
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.session_manager = session_manager
        
        # Load configs from environment with sensible defaults
        try:
            self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
        except ValueError:
            self.similarity_threshold = 0.55
            
        try:
            self.top_k = int(os.getenv("TOP_K", "3"))
        except ValueError:
            self.top_k = 3

        logger.info(f"RAGService initialized with similarity_threshold={self.similarity_threshold}, top_k={self.top_k}")

    def chunk_text(self, text: str, max_words: int = 150) -> List[str]:
        """Splits long text into chunks of maximum words while preserving whole sentences."""
        # Simple sentence splitter
        sentences = text.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|")
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > max_words:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def index_documents(self, docs_path: str):
        """Loads and indexes documents from a JSON file into the vector store."""
        if not os.path.exists(docs_path):
            logger.warning(f"Knowledge documents file {docs_path} does not exist. Skipping indexing.")
            return

        # Check if already indexed to prevent redundant API calls
        existing_count = self.vector_store.get_chunk_count()
        if existing_count > 0:
            logger.info(f"Vector store already contains {existing_count} chunks. Skipping re-indexing.")
            return

        logger.info(f"Loading documents from {docs_path} for indexing...")
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
                
            if not isinstance(documents, list):
                raise ValueError("docs.json must be a JSON array of documents.")

            for doc_idx, doc in enumerate(documents):
                title = doc.get("title", f"Document {doc_idx}")
                content = doc.get("content", "")
                
                if not content:
                    logger.warning(f"Document '{title}' has empty content. Skipping.")
                    continue

                # Chunk content
                chunks = self.chunk_text(content)
                logger.info(f"Document '{title}' split into {len(chunks)} chunk(s).")
                
                for chunk_idx, chunk_content in enumerate(chunks):
                    # Generate embedding
                    embedding = self.embedding_service.get_embedding(chunk_content, is_query=False)
                    
                    # Construct metadata
                    metadata = {
                        "chunk_id": f"{doc_idx}_{chunk_idx}",
                        "source_document": docs_path,
                        "total_chunks": len(chunks)
                    }
                    
                    # Store in vector store
                    self.vector_store.add_chunk(
                        doc_title=title,
                        content=chunk_content,
                        embedding=embedding,
                        metadata=metadata
                    )
            
            logger.info("Successfully indexed all document chunks.")
        except Exception as e:
            logger.error(f"Error occurred during indexing: {e}")
            raise e

    def process_query(self, session_id: str, query: str) -> Tuple[str, int, int]:
        """
        Main query pipeline:
        1. Embed query
        2. Perform similarity search
        3. Threshold filter (Grounding check)
        4. Assemble history and prompt
        5. Invoke LLM and save session history
        
        Returns:
            Tuple[response_reply, tokens_used, retrieved_chunks_count]
        """
        # 1. Generate query embedding
        query_emb = self.embedding_service.get_embedding(query, is_query=True)
        
        # 2. Retrieve Top-K matching chunks
        retrieved_chunks = self.vector_store.similarity_search(query_emb, top_k=self.top_k)
        
        # 3. Grounding check against similarity threshold
        highest_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
        
        if not retrieved_chunks or highest_score < self.similarity_threshold:
            logger.warning(f"Highest similarity score {highest_score:.4f} is below threshold {self.similarity_threshold}. Returning fallback.")
            fallback_msg = "I could not find enough information in the knowledge base to answer this question."
            
            # Even in fallback, we track the user question in session history
            self.session_manager.add_message(session_id, "user", query)
            self.session_manager.add_message(session_id, "assistant", fallback_msg)
            
            # 0 chunks successfully used, 0 LLM tokens used
            return fallback_msg, 0, 0

        # Filter chunks that pass threshold (to avoid injecting low-quality noise context)
        valid_chunks = [c for c in retrieved_chunks if c["score"] >= self.similarity_threshold]
        
        # 4. Build context from retrieved chunks
        context_parts = []
        for idx, chunk in enumerate(valid_chunks):
            context_parts.append(f"[{idx+1}] Source: {chunk['title']}\nContent: {chunk['content']}")
            
        context_str = "\n\n".join(context_parts)
        
        # Format history
        history_str = self.session_manager.get_history_string(session_id)
        
        # Compile prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            retrieved_context=context_str,
            history=history_str,
            user_question=query
        )
        
        # 5. Invoke LLM
        reply, tokens_used = self.llm_service.generate_response(prompt)
        
        # Clean up any potential markdown headers or instructions from the model
        reply_cleaned = reply.strip()
        
        # Save interaction to session memory
        self.session_manager.add_message(session_id, "user", query)
        self.session_manager.add_message(session_id, "assistant", reply_cleaned)
        
        return reply_cleaned, tokens_used, len(valid_chunks)
