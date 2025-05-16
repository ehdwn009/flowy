from pydantic import BaseModel, EmailStr, Field, HttpUrl # EmailStr, HttpUrl 등 유용한 타입 활용 가능
from typing import List, Optional, Dict, Any # Python 타이핑을 위한 모듈
from fastapi import UploadFile # 파일 업로드 타입

# --- 공통 모델 ---
class AttendeeModel(BaseModel):
    name: str = Field(..., description="참석자 이름")
    email: Optional[EmailStr] = Field(None, description="참석자 이메일 (선택 사항)")
    role: Optional[str] = Field(None, description="참석자 역할 (선택 사항)")

class MeetingInfoBase(BaseModel):
    subj: Optional[str] = Field(None, description="회의 주제")
    df: Optional[str] = Field(None, description="회의 일시 (YYYY-MM-DDTHH:MM:SS 형식 권장)")
    loc: Optional[str] = Field(None, description="회의 장소")
    # info_n 대신 attendees 로 명칭 변경 제안 (더 명확한 의미)
    attendees: Optional[List[AttendeeModel]] = Field(None, description="참석자 정보 리스트")

# --- STT (Speech-to-Text) 관련 모델 ---
# STT 요청 시 파일과 함께 다른 메타데이터를 어떻게 받을지에 따라 모델 구조가 달라질 수 있습니다.
# 여기서는 간단히 파일은 별도 파라미터로 받고, 메타데이터는 선택적으로 받는다고 가정합니다.
# 또는 모든 정보를 포함하는 하나의 요청 모델을 만들 수도 있습니다.

class STTResponse(BaseModel):
    rc_txt: str = Field(..., description="음성인식 변환 결과 텍스트")
    message: Optional[str] = Field(None, description="처리 결과 메시지")
    # 필요하다면 여기에 원본 파일명, 처리 시간 등 추가 정보 포함 가능

# --- 요약 (Summarization) 관련 모델 ---
class SummarizationRequest(MeetingInfoBase): # MeetingInfoBase 상속하여 공통 필드 사용
    rc_txt: str = Field(..., description="요약할 원본 텍스트 (회의록)")
    # subj는 MeetingInfoBase 에 이미 포함되어 있음

class SummaryPoint(BaseModel): # 요약 결과가 여러 불렛포인트일 경우를 위해
    point: str

class SummarizationResponse(BaseModel):
    # 프론트에서 'summary' 라는 키를 기대한다고 하셨으니, 그에 맞춰 구성
    # summarize_meeting 함수의 결과가 {"summary_points": ["요약1", "요약2"]} 형태이므로,
    # 이를 감안하여 모델 설계 또는 서비스 계층에서 변환 필요
    summary: List[str] = Field(..., description="회의 요약 결과 (불렛 포인트 리스트)")
    # 또는 summary: Dict[str, List[str]] = Field(..., description="회의 요약 결과 {'summary_points': [...]}")
    message: Optional[str] = Field(None, description="처리 결과 메시지")


# --- 역할 및 할 일 분배 (Action Item Assignment) 관련 모델 ---
class AssignedTask(BaseModel):
    task_description: str = Field(..., description="할 일 내용")
    # due_date: Optional[str] = Field(None, description="마감 기한 (LLM이 추출한 그대로)") # action_item_extractor.py 에서는 task 문자열에 기한 포함

class ActionItemByAssignee(BaseModel):
    name: str = Field(..., description="담당자 이름")
    role: Optional[str] = Field(None, description="담당자 역할")
    tasks: List[str] = Field(..., description="담당자별 할 일 목록 (문자열 리스트)") # 기존 코드에서는 문자열 리스트

class ActionAssignmentRequest(MeetingInfoBase): # MeetingInfoBase 상속
    rc_txt: str = Field(..., description="분석할 원본 텍스트 (회의록)")
    # subj, attendees 는 MeetingInfoBase 에 포함

class ActionAssignmentResponse(BaseModel):
    # 프론트에서 'tasks' 라는 키를 기대한다고 하셨으니, 그에 맞춰 구성
    tasks: List[ActionItemByAssignee] = Field(..., description="담당자별 할 일 목록")
    message: Optional[str] = Field(None, description="처리 결과 메시지")

# --- 회의 피드백 (Meeting Feedback / Relevance Analysis) 관련 모델 ---
class OverallStatisticsModel(BaseModel): # 기존 FeedbackStatistics 모델 이름 변경 및 필드명 수정
    sentences: int = Field(description="전체 문장 수")
    necessary: int = Field(description="필요 문장 수")
    unnecessary: int = Field(description="불필요 문장 수")
    error: int = Field(description="오류 문장 수")
    necessary_ratio: float = Field(description="필요 문장 비율 (%)")
    unnecessary_ratio: float = Field(description="불필요 문장 비율 (%)")
    error_ratio: float = Field(description="오류 문장 비율 (%)")

    # 만약 relevance_analyzer.py의 반환 키가 "전체_문장_수" 등 한글 키이고
    # 모델 필드는 영어로 하고 싶다면 alias를 사용합니다.
    # sentences: int = Field(alias="전체_문장_수", description="전체 문장 수")
    # necessary: int = Field(alias="필요_문장_수", description="필요 문장 수")
    # ...
    # class Config:
    #     populate_by_name = True # JSON 키 -> 모델 필드 매핑 시 alias 사용

class RepresentativeUnnecessarySentenceModel(BaseModel): # 기존 RepresentativeUnnecessarySentence 모델 이름 변경 (선택적)
    sentence: str = Field(description="대표적인 불필요 문장")
    reason: str = Field(description="불필요하다고 판단한 이유")

class MeetingFeedbackResponseModel(BaseModel):
    overall_statistics: OverallStatisticsModel = Field(description="회의록 문장 전체 통계")
    representative_unnecessary: List[RepresentativeUnnecessarySentenceModel] = Field(description="대표적인 불필요 문장 목록")
    message: Optional[str] = Field(None, description="처리 결과 메시지")
    rc_txt_splitted: Optional[List[str]] = Field(None, description="문장 분리 결과 (클라이언트 요청 시)")

    # 만약 서비스 로직(relevance_service.py)에서 반환하는 딕셔너리의 키가
    # "전체_통계", "대표_불필요_문장" 등이고,
    # API 응답 JSON 키는 "overall_statistics", "representative_unnecessary"로 하고 싶다면
    # Pydantic 모델 필드명은 JSON 키와 일치시키고,
    # 서비스 로직에서 데이터를 가져와 모델을 생성할 때 키 매핑을 해주거나,
    # 또는 모델 필드에 alias를 사용하여 서비스 로직의 키와 매핑할 수 있습니다.

    # 예시: 서비스 로직 반환 키 -> 모델 필드(JSON 키) 매핑 (alias 사용)
    # overall_statistics: OverallStatisticsModel = Field(alias="전체_통계", description="회의록 문장 전체 통계")
    # representative_unnecessary: List[RepresentativeUnnecessarySentenceModel] = Field(alias="대표_불필요_문장", description="대표적인 불필요 문장 목록")
    # class Config:
    #     populate_by_name = True # 데이터를 모델로 로드할 때 alias 사용


# Feedback 요청 모델은 이전과 거의 동일하게 유지할 수 있습니다.
class FeedbackRequest(MeetingInfoBase):
    rc_txt: str = Field(..., description="분석할 원본 텍스트 (회의록)")
    num_representative_unnecessary: Optional[int] = Field(5, description="대표 불필요 문장 개수")


# --- (추가 제안) 통합 분석 요청 및 응답 모델 ---
# 만약 하나의 API로 모든 것을 처리한다면 사용할 수 있는 모델
class FullAnalysisRequest(MeetingInfoBase):
    rc_file: Optional[UploadFile] = Field(None, description="녹음 파일 (m4a)")
    rc_txt: Optional[str] = Field(None, description="회의록 텍스트 (텍스트 직접 입력 시)")

    # Pydantic 모델의 root_validator를 사용하여 rc_file과 rc_txt 중 하나만 존재하도록 강제할 수 있음
    # from pydantic import root_validator
    # @root_validator(pre=True)
    # def check_input_source(cls, values):
    #     rc_file, rc_txt = values.get('rc_file'), values.get('rc_txt')
    #     if bool(rc_file) == bool(rc_txt): # 둘 다 있거나 둘 다 없으면 에러
    #         raise ValueError("rc_file 또는 rc_txt 중 하나만 제공되어야 합니다.")
    #     return values


class FullAnalysisResponse(BaseModel):
    meeting_info: MeetingInfoBase
    stt_result: Optional[STTResponse] = None
    summary_result: Optional[SummarizationResponse] = None # SummarizationResponse 모델도 정의되어 있어야 함
    action_items_result: Optional[ActionAssignmentResponse] = None # ActionAssignmentResponse 모델도 정의되어 있어야 함
    # feedback_result: Optional[MeetingFeedbackResponse] = None # <--- 여기가 문제!
    feedback_result: Optional[MeetingFeedbackResponseModel] = None # <--- 이렇게 수정!
    message: str = "분석이 완료되었습니다."