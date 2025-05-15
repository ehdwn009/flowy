from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form
from typing import Any, Optional # STT 파이프라인 객체 타입 힌팅 및 선택적 파라미터용

# 의존성 주입 함수 (main.py 에서 가져옴)
from app.main import get_stt_pipeline

# 서비스 함수 (stt_service.py 에서 가져옴)
from app.services.stt_service import process_uploaded_rc_file_to_text

# 응답 모델 (models/meeting.py 에서 가져옴)
from app.models.meeting import STTResponse

# --- 라우터 객체 생성 ---
# 이 라우터는 app.main.py에서 prefix="/api/stt", tags=["1. ..."] 로 포함될 예정
router = APIRouter()

# --- STT API 엔드포인트 정의 ---
@router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio_endpoint(
    # 파일 업로드 (필수)
    # 프론트엔드에서 'rc_file' 이라는 이름으로 파일을 보내야 합니다.
    rc_file: UploadFile = File(..., description="변환할 음성 파일 (m4a 등)"),

    # STT 처리 관련 선택적 파라미터 (Form 데이터로 받음)
    # services/stt_service.py 의 process_uploaded_rc_file_to_text 함수 시그니처와 일치시킴
    target_language: Optional[str] = Form("ko", description="음성 인식 대상 언어 (예: 'ko', 'en')"),
    pipeline_chunk_length_s: Optional[int] = Form(30, description="파이프라인 청크 길이 (초)"),
    pipeline_stride_length_s: Optional[int] = Form(5, description="파이프라인 스트라이드 길이 (초, 오버랩 관련)"),
    # 여기에 추가적인 메타데이터 (예: 회의 주제, 참석자 정보 등)를 Form 데이터로 받을 수 있습니다.
    # subj: Optional[str] = Form(None, description="회의 주제 (선택 사항)"),
    # df: Optional[str] = Form(None, description="회의 일시 (선택 사항)"),
    # ...

    # 의존성 주입: 미리 로드된 STT 파이프라인 가져오기
    stt_pipeline: Any = Depends(get_stt_pipeline)
):
    """
    업로드된 음성 파일 (`rc_file`)을 텍스트로 변환합니다.
    """
    if not rc_file.content_type or not (
        rc_file.content_type.startswith("audio/") or
        rc_file.content_type == "application/octet-stream" or # m4a가 이렇게 올 수도 있음
        rc_file.content_type == "video/mp4" # m4a가 mp4 컨테이너 사용
    ):
        # 간단한 MIME 타입 체크 (더 엄격하게 할 수도 있음)
        print(f"STT 라우터 경고: 예상치 못한 파일 타입 수신 - '{rc_file.filename}' (타입: {rc_file.content_type})")
        # raise HTTPException(
        #     status_code=400,
        #     detail=f"지원하지 않는 파일 형식입니다: {rc_file.content_type}. 오디오 파일을 업로드해주세요."
        # )
        # 우선 경고만 하고 진행 (m4a 등 다양한 오디오 컨테이너 고려)

    print(f"STT 라우터: 파일 수신 시작 - '{rc_file.filename}', 언어: {target_language}")

    try:
        # 서비스 함수 호출하여 STT 처리
        transcribed_text = await process_uploaded_rc_file_to_text(
            rc_file=rc_file,
            stt_pipeline_instance=stt_pipeline,
            target_language=target_language,
            pipeline_chunk_length_s=pipeline_chunk_length_s,
            pipeline_stride_length_s=pipeline_stride_length_s
            # 추가적인 메타데이터가 있다면 여기에 전달
            # subj=subj,
            # df=df,
        )

        if transcribed_text is None or transcribed_text.strip() == "":
            # STT 결과가 비어있는 경우 (예: 묵음 구간만 있는 파일)
            print(f"STT 라우터: '{rc_file.filename}' 파일에서 변환된 텍스트가 없습니다.")
            # 클라이언트 요구사항에 따라 오류로 처리하거나, 빈 텍스트로 정상 응답할 수 있음
            # 여기서는 빈 텍스트도 정상 응답으로 간주
            return STTResponse(rc_txt="", message="음성 파일에서 인식된 텍스트가 없습니다.")

        print(f"STT 라우터: '{rc_file.filename}' 변환 성공.")
        return STTResponse(
            rc_txt=transcribed_text,
            message="오디오 파일이 성공적으로 텍스트로 변환되었습니다."
        )

    except FileNotFoundError as e_fnf:
        # 서비스 계층에서 발생한 예외를 잡아서 적절한 HTTP 응답으로 변환
        print(f"STT 라우터 오류 (FileNotFoundError): {e_fnf}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: 파일 처리 중 문제가 발생했습니다. ({e_fnf})")
    except ValueError as e_val:
        print(f"STT 라우터 오류 (ValueError): {e_val}")
        raise HTTPException(status_code=400, detail=f"입력 데이터 오류: {e_val}")
    except RuntimeError as e_rt: # 모델 실행 관련 오류 등
        print(f"STT 라우터 오류 (RuntimeError): {e_rt}")
        raise HTTPException(status_code=503, detail=f"서비스 처리 중 오류가 발생했습니다: {e_rt}")
    except HTTPException as e_http:
        # 이미 HTTPException인 경우 그대로 다시 발생
        raise e_http
    except Exception as e:
        # 예상치 못한 모든 오류 처리
        print(f"STT 라우터 알 수 없는 오류: {type(e).__name__} - {e}")
        # traceback.print_exc() # 개발 중 스택 트레이스 출력
        raise HTTPException(status_code=500, detail="알 수 없는 서버 오류가 발생했습니다.")