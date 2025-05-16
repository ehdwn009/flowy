# app/models/meeting.py
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Any
from fastapi import UploadFile # <--- 이 줄을 추가해주세요!

# --- 공통 모델 ---
class AttendeeInfo(BaseModel):
    name: str = Field(..., description="참여자 이름")
    email: Optional[EmailStr] = Field(None, description="참여자 이메일 (선택 사항)")
    role: Optional[str] = Field(None, description="참여자 역할 (선택 사항)")

class MeetingInfoBase(BaseModel):
    subj: Optional[str] = Field(None, description="회의 주제")
    df: Optional[str] = Field(None, description="회의 일시 (YYYY-MM-DDTHH:MM:SS 형식 권장)")
    loc: Optional[str] = Field(None, description="회의 장소")
    info_n: Optional[List[AttendeeInfo]] = Field(None, description="참석자 정보 리스트 (기존 'info_n' 사용)")

# --- STT 관련 모델 ---
class STTResponse(BaseModel):
    rc_txt: str = Field(..., description="음성인식 변환 결과 텍스트")
    message: Optional[str] = Field(None, description="처리 결과 메시지")

# --- 요약 (Summarization) 관련 모델 ---
class SummarizationRequest(BaseModel):
    rc_txt: str = Field(..., description="요약할 원본 텍스트 (회의록)")
    subj: Optional[str] = Field(None, description="회의 주제 (선택 사항, 요약 품질 향상에 도움)")

class SummarizationResponse(BaseModel):
    summary: List[str] = Field(..., description="회의 요약 결과 (불렛 포인트 리스트)")
    message: Optional[str] = Field(None, description="처리 결과 메시지")

# --- 역할 및 할 일 분배 (Action Item Assignment) 관련 모델 ---
class ActionItemByAssignee(BaseModel):
    name: str = Field(..., description="담당자 이름")
    role: Optional[str] = Field(None, description="담당자 역할")
    tasks: List[str] = Field(..., description="담당자별 할 일 목록 (문자열 리스트)")

class ActionAssignmentRequest(BaseModel):
    rc_txt: str = Field(..., description="분석할 원본 텍스트 (회의록)")
    subj: Optional[str] = Field(None, description="회의 주제 (선택 사항)")
    info_n: List[AttendeeInfo] = Field(..., description="참석자 정보 리스트 (필수)")

class ActionAssignmentResponse(BaseModel):
    tasks: List[ActionItemByAssignee] = Field(..., description="담당자별 할 일 목록")
    message: Optional[str] = Field(None, description="처리 결과 메시지")

# --- 회의 피드백 (Meeting Feedback / Relevance Analysis) 관련 모델 ---
class OverallStatisticsModel(BaseModel):
    sentences: int = Field(description="전체 문장 수")
    necessary: int = Field(description="필요 문장 수")
    unnecessary: int = Field(description="불필요 문장 수")
    error: int = Field(description="오류 문장 수")
    necessary_ratio: float = Field(description="필요 문장 비율 (%)")
    unnecessary_ratio: float = Field(description="불필요 문장 비율 (%)")
    error_ratio: float = Field(description="오류 문장 비율 (%)")

class RepresentativeUnnecessarySentenceModel(BaseModel):
    sentence: str = Field(description="대표적인 불필요 문장")
    reason: str = Field(description="불필요하다고 판단한 이유")

class MeetingFeedbackResponseModel(BaseModel):
    overall_statistics: OverallStatisticsModel
    representative_unnecessary: List[RepresentativeUnnecessarySentenceModel]
    message: Optional[str] = Field(None, description="처리 결과 메시지")
    rc_txt_splitted: Optional[List[str]] = Field(None, description="문장 분리 결과 (클라이언트 요청 시)")

class FeedbackRequest(BaseModel):
    rc_txt: str = Field(..., description="분석할 원본 텍스트 (회의록)")
    subj: Optional[str] = Field(None, description="회의 주제 (선택 사항)")
    info_n: List[AttendeeInfo] = Field(..., description="참석자 정보 리스트 (필수)")
    num_representative_unnecessary: Optional[int] = Field(5, description="대표 불필요 문장 개수")

# --- 통합 분석 요청 및 응답 모델 (필요시 사용) ---
class FullAnalysisRequest(MeetingInfoBase):
    rc_file: Optional[UploadFile] = Field(None, description="녹음 파일 (m4a)") # 여기서 UploadFile 사용
    rc_txt: Optional[str] = Field(None, description="회의록 텍스트 (텍스트 직접 입력 시)")

class FullAnalysisResponse(BaseModel):
    meeting_info: MeetingInfoBase
    stt_result: Optional[STTResponse] = None
    summary_result: Optional[SummarizationResponse] = None
    action_items_result: Optional[ActionAssignmentResponse] = None
    feedback_result: Optional[MeetingFeedbackResponseModel] = None
    message: str = "분석이 완료되었습니다."