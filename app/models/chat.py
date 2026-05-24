from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    sessionId: str = Field(default="default_session", description="Session identifier for tracking history")
    message: str = Field(..., description="The query message from the user")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sessionId": "abc123",
                "message": "How can I reset my password?"
            }
        }
    }

class ChatResponse(BaseModel):
    reply: str = Field(..., description="The assistant's grounded response")
    tokensUsed: int = Field(..., description="Approximate number of API tokens consumed")
    retrievedChunks: int = Field(..., description="Number of document chunks retrieved for context")
