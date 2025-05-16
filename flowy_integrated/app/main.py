# app/main.py
from fastapi import FastAPI, Depends, HTTPException # Depends, HTTPException 추가
from typing import Any, Optional, List # List 추가
from openai import OpenAI # OpenAI 임포트
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq # STT용
import torch # STT용
import os # (선택적, 경로 관련 로직에 필요시)

from app.core.config import settings
from app.routers import analysis # , notification (나중에 추가)

# --- FastAPI 앱 인스턴스 생성 ---
app = FastAPI(
    title="Flowy 회의록 분석 및 알림 API",
    description="음성 또는 텍스트로 입력된 회의록을 분석하고, 결과를 이메일로 알림을 보낼 수 있는 API입니다.",
    version="0.1.0"
)

# --- 의존성 주입 (Dependency Injection) 전역 변수 ---
_openai_client: Optional[OpenAI] = None
_stt_pipeline_instance: Optional[Any] = None # 실제 파이프라인 타입으로 변경 가능
_stt_model_name = "openai/whisper-large-v3" # 사용할 Whisper 모델 (변경 가능, config에서 관리 추천)

# --- 의존성 주입 함수들 ---
def get_openai_client():
    """OpenAI 클라이언트 인스턴스를 반환합니다. 없으면 생성합니다."""
    global _openai_client
    if _openai_client is None:
        if not settings.OPENAI_API_KEY:
            print("오류: OPENAI_API_KEY가 설정되지 않았습니다. (main.py)")
            raise HTTPException(status_code=500, detail="OpenAI API 키가 설정되지 않았습니다.")
        try:
            _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            print("OpenAI 클라이언트가 성공적으로 초기화되었습니다. (main.py)")
        except Exception as e:
            print(f"OpenAI 클라이언트 초기화 실패: {e} (main.py)")
            raise HTTPException(status_code=500, detail=f"OpenAI 클라이언트 초기화 중 오류 발생: {str(e)}")
    return _openai_client

def initialize_stt_pipeline():
    """
    Hugging Face Transformers STT 파이프라인을 초기화하고 로드합니다.
    """
    global _stt_pipeline_instance
    # 이미 초기화되었으면 다시 시도하지 않음
    if _stt_pipeline_instance is not None:
        return _stt_pipeline_instance
        
    try:
        print(f"STT 파이프라인 ({_stt_model_name}) 초기화 시도... (main.py)")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"STT 파이프라인을 위한 장치: {device} (main.py)")

        # Whisper 모델 로드 (예시, 필요에 따라 tokenizer, feature_extractor 등 명시적 로드)
        _stt_pipeline_instance = hf_pipeline(
            "automatic-speech-recognition",
            model=_stt_model_name,
            device=device,
            # chunk_length_s=30, # 긴 오디오 처리에 필요할 수 있음
            # stride_length_s=[5, 5], # 청크 간 겹침 설정
        )
        print(f"STT 파이프라인 ({_stt_model_name}) 초기화 성공. (main.py)")
    except ImportError:
        print("STT 관련 패키지(transformers, torch 등)가 설치되지 않았습니다. STT 기능을 사용할 수 없습니다. (main.py)")
        _stt_pipeline_instance = None # 명시적으로 None 설정
    except Exception as e:
        print(f"STT 파이프라인 ({_stt_model_name}) 초기화 실패: {e} (main.py)")
        _stt_pipeline_instance = None
    return _stt_pipeline_instance

def get_stt_pipeline():
    """STT 파이프라인 인스턴스를 반환합니다. 없으면 초기화를 시도합니다."""
    if _stt_pipeline_instance is None:
        return initialize_stt_pipeline()
    return _stt_pipeline_instance

# --- 애플리케이션 이벤트 핸들러 (Lifespan 추천) ---
# @app.on_event("startup") # Deprecated, lifespan 사용 권장
# async def startup_event():
# print("애플리케이션 시작 이벤트: 초기화 시도 (main.py)")
# initialize_stt_pipeline()
# get_openai_client() # OpenAI 클라이언트도 미리 초기화 시도

# FastAPI Lifespan 사용 예시 (Python 3.9+ async context manager)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # 애플리케이션 시작 시 실행될 코드
    print("애플리케이션 시작 (lifespan): 초기화 시도 (main.py)")
    initialize_stt_pipeline() # STT 파이프라인 미리 로드
    get_openai_client()       # OpenAI 클라이언트 미리 초기화 (선택적)
    yield
    # 애플리케이션 종료 시 실행될 코드 (필요시)
    print("애플리케이션 종료 (lifespan). (main.py)")

# app 인스턴스 생성 시 lifespan 적용
app = FastAPI(
    title="Flowy 회의록 분석 및 알림 API",
    description="음성 또는 텍스트로 입력된 회의록을 분석하고, 결과를 이메일로 알림을 보낼 수 있는 API입니다.",
    version="0.1.0",
    lifespan=lifespan # lifespan 이벤트 핸들러 등록
)

# --- API 라우터 등록 ---
app.include_router(analysis.router, prefix="/api/v1", tags=["Meeting Analysis"])
# app.include_router(notification.router, prefix="/api/v1", tags=["Notifications"]) # 추후 추가

# --- 기본 루트 엔드포인트 ---
@app.get("/", tags=["Root"])
async def read_root():
    """API의 루트 엔드포인트입니다. 서비스 상태를 간단히 확인할 수 있습니다."""
    return {"message": "Flowy 회의록 분석 API에 오신 것을 환영합니다! (v0.1.0)"}