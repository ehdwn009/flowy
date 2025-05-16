# app/models/meeting.py
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Any

# --- 공통 모델 ---
class AttendeeInfo(BaseModel):
    name: str = Field(..., description="참여자 이름")
    email: Optional[EmailStr] = Field(None, description="참여자 이메일 (선택 사항)")
    role: Optional[str] = Field(None, description="참여자 역할 (선택 사항)")

# --- Form으로 받을 JSON 문자열 내부 구조를 위한 모델 ---
class MeetingMetadata(BaseModel):
    subj: str = Field(..., description="회의 주제")
    dt: str = Field(..., description="회의 일시 (YYYY-MM-DDTHH:MM:SS 형식 권장)")
    loc: str = Field(..., description="회의 장소")
    info_n: List[AttendeeInfo] = Field(..., description="참석자 정보 리스트 (실제 객체 리스트)")

# --- STT 관련 모델 ---
class STTResponse(BaseModel):
    rc_txt: str = Field(..., description="음성인식 변환 결과 텍스트")
    message: Optional[str] = Field(None, description="처리 결과 메시지")

# --- 요약 (Summarization) 관련 모델 ---
class SummarizationResponse(BaseModel):
    summary: List[str] = Field(..., description="회의 요약 결과 (불렛 포인트 리스트)")
    message: Optional[str] = Field(None, description="처리 결과 메시지")

# --- 역할 및 할 일 분배 (Action Item Assignment) 관련 모델 ---
class ActionItemByAssignee(BaseModel):
    name: str = Field(..., description="담당자 이름")
    role: Optional[str] = Field(None, description="담당자 역할")
    tasks: List[str] = Field(..., description="담당자별 할 일 목록 (문자열 리스트)")

class ActionAssignmentResponse(BaseModel):
    tasks: List[ActionItemByAssignee] = Field(..., description="담당자별 할 일 목록")
    message: Optional[str] = Field(None, description="처리 결과 메시지")

# --- 회의 피드백 (Meeting Feedback / Relevance Analysis) 관련 모델 ---
class RepresentativeUnnecessarySentenceModel(BaseModel):
    sentence: str = Field(..., description="대표적인 불필요 문장")
    reason: str = Field(..., description="불필요하다고 판단한 이유")

class MeetingFeedbackResponseModel(BaseModel):
    necessary_ratio: float = Field(..., description="필요 문장 비율 (%)")
    unnecessary_ratio: float = Field(..., description="불필요 문장 비율 (%)")
    representative_unnecessary: List[RepresentativeUnnecessarySentenceModel] = Field(
        default_factory=list, description="대표적인 불필요 문장 목록 (최대 5개)"
    )

# --- 통합 분석 결과 모델 ---
class FullAnalysisResult(BaseModel):
    meeting_info: MeetingMetadata # MeetingMetadata 사용
    # stt_result: Optional[STTResponse] = None # STT 결과는 포함 (필요시 제외 가능)
    summary_result: SummarizationResponse
    action_items_result: ActionAssignmentResponse
    feedback_result: MeetingFeedbackResponseModel