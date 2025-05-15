from fastapi import APIRouter, Depends
from openai import OpenAI
from app.main import get_openai_client
# from app.models.meeting import FeedbackRequest, MeetingFeedbackResponseModel
# from app.services.relevance_service import ...
# from app.services.text_process_service import ...

router = APIRouter()

@router.post("/analyze-sentences", response_model=dict)
async def analyze_sentences_endpoint(
    text_for_feedback: str, # 임시
    openai_client: OpenAI = Depends(get_openai_client)
):
    return {"feedback": f"'{text_for_feedback}'에 대한 문장 분석 피드백 (구현 예정)"}