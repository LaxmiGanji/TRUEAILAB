from fastapi import APIRouter, Request, HTTPException, status
from app.models.chat import ChatRequest, ChatResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, payload: ChatRequest):
    """
    POST /api/chat
    Accepts sessionId and message, retrieves context, queries the LLM, 
    and logs similarity scores.
    """
    # Retrieve the RAG service instance from application state
    rag_service = getattr(request.app.state, "rag_service", None)
    if not rag_service:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG orchestration service is not initialized on the server."
        )

    # 1. Validation check (ensure message is not empty)
    message_text = payload.message.strip() if payload.message else ""
    if not message_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field is required and cannot be empty."
        )

    logger.info(f"Received chat request. SessionId: {payload.sessionId} | Message: '{message_text[:40]}...'")

    try:
        # 2. Retrieve, ground, build context, and invoke LLM
        reply, tokens_used, retrieved_chunks = rag_service.process_query(
            session_id=payload.sessionId,
            query=message_text
        )

        return ChatResponse(
            reply=reply,
            tokensUsed=tokens_used,
            retrievedChunks=retrieved_chunks
        )

    except ValueError as e:
        logger.error(f"Config/Validation Error in Chat pipeline: {e}")
        # Map ValueErrors to 400 Bad Request (such as missing/invalid API Key)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except TimeoutError as e:
        logger.error(f"LLM API Timeout Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The AI generation request timed out. Please try again."
        )
    except RuntimeError as e:
        logger.error(f"Runtime Error in Chat pipeline: {e}")
        # Catch Rate Limit Exceeded or general service exceptions
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unhandled Exception in Chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {str(e)}"
        )
