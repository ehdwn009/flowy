# app/routers/analysis.py
import json
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from typing import Optional, List, Any
from openai import OpenAI

# main.py 에서 정의한 의존성 주입 함수들
from app.main import get_openai_client, get_stt_pipeline

# 설정값 (LLM 모델명 등)
from app.core.config import settings

# Pydantic 모델들
from app.models.meeting import (
    AttendeeInfo, # MeetingMetadata 내부에서 사용
    MeetingMetadata, # Form으로 받을 JSON 문자열을 파싱할 모델
    STTResponse,
    SummarizationResponse, ActionItemByAssignee, ActionAssignmentResponse,
    MeetingFeedbackResponseModel, RepresentativeUnnecessarySentenceModel,
    FullAnalysisResult
)

# 서비스 함수들
from app.services.stt_service import process_uploaded_rc_file_to_text
from app.services.summarizer_service import get_meeting_summary
from app.services.action_item_service import extract_and_assign_action_items
from app.services.relevance_service import analyze_sentence_relevance_service

router = APIRouter()

@router.post("/analyze", response_model=FullAnalysisResult)
async def analyze_meeting_endpoint(
    # 프론트엔드에서 회의 정보를 담은 JSON 문자열을 'metadata_json'이라는 이름의 Form 필드로 보낸다고 가정
    metadata_json: str = Form(..., description="회의 주제, 일시, 장소, 참석자 정보(배열)를 포함하는 JSON 문자열"),
    rc_file: Optional[UploadFile] = File(None, description="녹음 파일 (m4a, wav 등)"),
    # rc_txt는 현재 시나리오에서 프론트가 보내지 않는다고 하셨으므로, 선택적으로 유지하거나 제거 가능
    # rc_txt: Optional[str] = Form(None, description="회의록 텍스트 (rc_file 없을 시 사용)"),
    openai_client: OpenAI = Depends(get_openai_client),
    stt_pipeline: Optional[Any] = Depends(get_stt_pipeline)
):
    """
    '분석하기' API 엔드포인트입니다.
    회의 정보(JSON 문자열)와 음성 파일을 입력받아 전체 분석을 수행합니다.
    """
    # 1. 메타데이터 JSON 문자열 파싱
    try:
        metadata_dict = json.loads(metadata_json)
        # Pydantic 모델을 사용하여 유효성 검사 및 객체화
        meeting_info_data = MeetingMetadata(**metadata_dict)
        print(f"분석 요청 수신 (메타데이터 파싱 성공): 주제='{meeting_info_data.subj}'")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="제공된 메타데이터가 유효한 JSON 형식이 아닙니다.")
    except Exception as e: # Pydantic ValidationError 등
        raise HTTPException(status_code=400, detail=f"메타데이터 처리 오류: {str(e)}")

    if rc_file:
        print(f"녹음 파일: {rc_file.filename if rc_file else '없음'}")
    # elif rc_txt: # rc_txt를 사용하지 않는다면 이 부분은 필요 없음
    #     print(f"텍스트 입력 길이: {len(rc_txt) if rc_txt else '0'}")


    # 2. 회의록 텍스트(current_rc_txt) 준비
    current_rc_txt = ""
    stt_response_data: Optional[STTResponse] = None
    # rc_txt를 프론트에서 받지 않으므로, 해당 로직은 rc_file에만 의존하거나,
    # rc_txt가 API 명세에 남아있다면 그에 대한 처리도 유지할 수 있습니다.
    # 여기서는 rc_txt를 사용하지 않는 것으로 가정하고 간소화.

    if rc_file:
        print("녹음 파일 처리 시작...")
        if not stt_pipeline:
            raise HTTPException(status_code=503, detail="STT 서비스를 현재 사용할 수 없습니다.")
        try:
            transcribed_text = await process_uploaded_rc_file_to_text(
                rc_file=rc_file,
                stt_pipeline_instance=stt_pipeline
            )
            if transcribed_text is None or not transcribed_text.strip():
                message = "음성 파일에서 텍스트를 추출하지 못했거나 내용이 없습니다."
                print(message)
                current_rc_txt = ""
                stt_response_data = STTResponse(rc_txt="", message=message)
            else:
                current_rc_txt = transcribed_text
                stt_response_data = STTResponse(rc_txt=current_rc_txt, message="음성 파일이 성공적으로 텍스트로 변환되었습니다.")
                print(f"STT 변환 완료. 텍스트 길이: {len(current_rc_txt)}")
        except Exception as e:
            print(f"STT 처리 중 오류 발생: {e}")
            raise HTTPException(status_code=500, detail=f"STT 처리 중 서버 오류 발생: {str(e)}")
    # elif rc_txt: # rc_txt를 사용하지 않으면 이 블록 제거
    #     print("제공된 텍스트 사용.")
    #     current_rc_txt = rc_txt.strip()
    #     if not current_rc_txt:
    #          stt_response_data = STTResponse(rc_txt="", message="제공된 텍스트가 비어있습니다.")
    #     else:
    #         stt_response_data = STTResponse(rc_txt=current_rc_txt, message="제공된 텍스트를 사용합니다.")
    else: # rc_file도 없고, (만약 rc_txt도 고려 안 한다면) 분석할 소스가 없음
        if not current_rc_txt: # current_rc_txt가 결국 비어있다면 (rc_file이 없거나 STT 실패)
            raise HTTPException(status_code=400, detail="분석할 음성 파일이 제공되지 않았습니다.")


    # 회의록 텍스트가 최종적으로 비어있는지 확인 (STT 실패 등)
    if not current_rc_txt and rc_file: # rc_file은 있었으나 STT 결과가 없는 경우
        print("STT 결과가 비어있어 분석을 진행할 수 없습니다.")
        # 이 경우 stt_response_data는 이미 메시지를 포함하고 있을 것임
        return FullAnalysisResult(
            meeting_info=meeting_info_data,
            stt_result=stt_response_data,
            summary_result=SummarizationResponse(summary=[], message="STT 결과가 없어 요약을 생성할 수 없습니다."),
            action_items_result=ActionAssignmentResponse(tasks=[], message="STT 결과가 없어 할 일을 추출할 수 없습니다."),
            feedback_result=MeetingFeedbackResponseModel(
                necessary_ratio=0.0, unnecessary_ratio=0.0, representative_unnecessary=[]
            )
        )

    # 3. 요약 (Summarization)
    print("회의 요약 시작...")
    try:
        summary_points = await get_meeting_summary(
            openai_client=openai_client,
            rc_txt=current_rc_txt,
            subj=meeting_info_data.subj, # 이제 meeting_info_data에서 가져옴
            model_name=settings.DEFAULT_LLM_MODEL
        )
        summary_result_data = SummarizationResponse(summary=summary_points, message="회의 내용이 성공적으로 요약되었습니다.")
    except Exception as e:
        summary_result_data = SummarizationResponse(summary=[], message=f"요약 생성 중 오류가 발생했습니다: {str(e)}")

    # 4. 할 일 분배 (Action Item Assignment)
    print("할 일 분배 시작...")
    # meeting_info_data.info_n 은 이미 List[AttendeeInfo] 이므로, model_dump() 사용
    attendees_list_for_service = [att.model_dump(exclude_none=True) for att in meeting_info_data.info_n]
    try:
        assigned_tasks_structured = await extract_and_assign_action_items(
            openai_client=openai_client,
            rc_txt=current_rc_txt,
            subj=meeting_info_data.subj,
            info_n=attendees_list_for_service,
            model_name=settings.DEFAULT_LLM_MODEL
        )
        action_items_result_data = ActionAssignmentResponse(
            tasks=[ActionItemByAssignee(**task_dict) for task_dict in assigned_tasks_structured],
            message="할 일이 성공적으로 추출 및 분배되었습니다." if assigned_tasks_structured else "추출된 할 일이 없습니다."
        )
    except Exception as e:
        action_items_result_data = ActionAssignmentResponse(tasks=[], message=f"할 일 추출 중 오류가 발생했습니다: {str(e)}")

    # 5. 회의 피드백 (Meeting Feedback / Relevance Analysis)
    print("회의 피드백 분석 시작...")
    try:
        feedback_result_dict = await analyze_sentence_relevance_service(
            openai_client=openai_client,
            rc_txt=current_rc_txt,
            subj=meeting_info_data.subj,
            info_n=attendees_list_for_service,
            model_name=settings.DEFAULT_LLM_MODEL
        )
        feedback_result_data = MeetingFeedbackResponseModel(
            necessary_ratio=feedback_result_dict.get("necessary_ratio", 0.0),
            unnecessary_ratio=feedback_result_dict.get("unnecessary_ratio", 0.0),
            representative_unnecessary=[
                RepresentativeUnnecessarySentenceModel(**item)
                for item in feedback_result_dict.get("representative_unnecessary", [])
            ]
        )
    except Exception as e:
        feedback_result_data = MeetingFeedbackResponseModel(
            necessary_ratio=0.0, unnecessary_ratio=0.0, representative_unnecessary=[]
        )

    print("모든 분석 완료.")
    return FullAnalysisResult(
        meeting_info=meeting_info_data, # 파싱된 MeetingMetadata 객체
        stt_result=stt_response_data,
        summary_result=summary_result_data,
        action_items_result=action_items_result_data,
        feedback_result=feedback_result_data
    )