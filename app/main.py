import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

# Load environment variables from .env
load_dotenv()

from app.utils.logger import setup_logger
from app.utils.session import SessionManager
from app.vectorstore.simple_store import SimpleVectorStore
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.rag import RAGService
from app.routes.chat import router as chat_router

# Initialize Logging
setup_logger()
logger = logging.getLogger("app.main")

app = FastAPI(
    title="TRUEAILAB GenAI Assistant with RAG",
    description="Production-grade chat assistant utilizing FastAPI, SQLite vector similarity, and Gemini LLM.",
    version="1.0.0"
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Validation Exception Handler to return flat error format: {"error": "details"}
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if not errors:
        return JSONResponse(status_code=400, content={"error": "Invalid request payload"})

    # Extract field and issue
    first_err = errors[0]
    loc_path = first_err.get("loc", [])
    field_name = loc_path[-1] if loc_path else "field"
    err_msg = first_err.get("msg", "validation error")
    
    # Standardize error message according to assignment requirement
    if field_name == "message" and ("missing" in err_msg or "required" in err_msg):
        custom_msg = "Message field is required"
    else:
        custom_msg = f"{field_name}: {err_msg}"
        
    logger.warning(f"Validation failed for request: {custom_msg}")
    return JSONResponse(
        status_code=400,
        content={"error": custom_msg}
    )

# Instantiate core services
db_path = os.getenv("VECTOR_DB_PATH", "vectorstore.db")
vector_store = SimpleVectorStore(db_path=db_path)
embedding_service = EmbeddingService()
llm_service = LLMService()
session_manager = SessionManager(max_pairs=5)

rag_service = RAGService(
    vector_store=vector_store,
    embedding_service=embedding_service,
    llm_service=llm_service,
    session_manager=session_manager
)

# Store rag_service on app state for access in routes
app.state.rag_service = rag_service

@app.on_event("startup")
async def startup_event():
    """Startup event to trigger indexing of the docs.json document base."""
    logger.info("Starting up FastAPI application...")
    docs_json_path = os.getenv("DOCS_JSON_PATH", "docs.json")
    try:
        logger.info(f"Triggering initial indexing on startup from '{docs_json_path}'")
        rag_service.index_documents(docs_json_path)
        logger.info(f"Startup indexing completed. Vector database size: {vector_store.get_chunk_count()} chunks.")
    except Exception as e:
        logger.error(f"Failed to perform startup indexing: {e}")
        logger.warning("The application will proceed, but search queries might fail until documents are indexed.")

# 1. Health Check Endpoint
@app.get("/health")
async def health_endpoint():
    """
    GET /health
    Returns service health status
    """
    return {"status": "healthy"}

# 2. API Routes
app.include_router(chat_router, prefix="/api")

# 3. Static Files Mounting for Frontend
# Mount the frontend directory to serve UI pages.
# Ensure this is mounted AFTER the api routes to avoid overlap.
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    logger.info(f"Mounted frontend static directory: {frontend_dir}")
else:
    logger.warning(f"Frontend directory '{frontend_dir}' was not found. Statically served pages will be unavailable.")
