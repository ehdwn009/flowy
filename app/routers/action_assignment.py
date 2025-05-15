from fastapi import APIRouter, Depends
from openai import OpenAI
from app.main import get_openai_client
# from app.models.meeting import ActionAssignmentRequest, ActionAssignmentResponse
# from app.services.action_item_service import ...

router = APIRouter()

@router.post("/assign-tasks", response_model=dict)
async def assign_tasks_endpoint(
    text_for_assignment: str, # 임시
    openai_client: OpenAI = Depends(get_openai_client)
):
    return {"tasks": f"'{text_for_assignment}' 기반 할 일 분배 결과 (구현 예정)"}