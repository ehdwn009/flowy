# app/services/stt_service.py

from fastapi import UploadFile
import shutil
import os
import uuid
import time
import re
import asyncio
from typing import Any, Optional

# --- 패키지 로드 및 가용성 확인 ---
_PACKAGES_AVAILABLE = False
_PIPELINE_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    from transformers.pipelines.audio_utils import ffmpeg_read
    from pydub import AudioSegment # 오디오 속성 확인 및 간단한 전처리에 사용 가능
    # import torchaudio # 또는 librosa, soundfile 등 오디오 로딩 라이브러리
    # import torch # pipeline 사용 시 명시적 torch 임포트는 필수는 아님
    _PACKAGES_AVAILABLE = True
    _PIPELINE_AVAILABLE = True # 파이프라인 사용 가능 플래그
    print("STT 서비스: 필수 패키지 (transformers, pydub) 로드 성공.")
except ImportError:
    print("STT 서비스 경고: 필수 패키지 (transformers, pydub 등)를 찾을 수 없습니다. STT 기능이 제한됩니다.")
    # 더미 클래스/함수 정의 (AttributeError 방지용)
    class AudioSegment:
        @staticmethod
        def from_file(dummy, format=None): return AudioSegment()
        def duration_seconds(self): return 0
        # 필요한 다른 더미 메소드 추가 가능

    def ffmpeg_read(bpayload: bytes, sampling_rate: int): # 더미 함수
        return b'' # 빈 바이트 반환


# --- 임시 저장 폴더 설정 ---
TEMP_UPLOAD_DIR = "temp_audio_uploads"
if not os.path.exists(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


# === 텍스트 후처리 유틸리티 ===

def _normalize_whitespace_and_punctuation(text: str) -> str:
    """일반적인 공백 및 구두점 정리"""
    if not text: return ""
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 구두점 앞 공백 제거 및 뒤 공백 추가 (이미 있다면 유지)
    text = re.sub(r'\s*([.,?!])', r'\1 ', text)
    # 연속된 구두점 정리 (예: "!! " -> "! ") - 간단한 처리
    text = re.sub(r'([.,?!])\1+', r'\1', text)
    # 문장 끝이 아닌 곳의 불필요한 공백 제거
    text = re.sub(r'\s+([.,?!])', r'\1', text) # 예: "단어 ." -> "단어."
    return text.strip()

def _remove_basic_repetitions(text: str, min_repeat_len: int = 3, max_repeat_times: int = 2) -> str:
    """
    단순 반복 단어/짧은 구문 제거 (예: "네 네 네", "그래서 그래서")
    min_repeat_len: 반복으로 간주할 최소 단어/토큰 길이
    max_repeat_times: 이 횟수 이상 반복될 경우 첫 번째만 남김
    """
    if not text: return ""
    # 정규식이 복잡해질 수 있으므로, 여기서는 간단한 로직으로 대체하거나,
    # 기존 stt.py의 정규식을 참고하여 적용 가능
    # 예: "네 네 네" -> "네"
    # 기존: re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1', filtered_text) # 3번 이상 반복 시 하나로
    # 조금 더 유연하게: max_repeat_times 이상 반복되는 경우
    # 이 부분은 성능과 정확도를 고려하여 더 정교한 알고리즘으로 대체 가능
    # 여기서는 아이디어만 제시하고, 기존 stt.py의 정규식을 활용하는 것을 고려
    
    # 기존 stt.py의 정규식 활용 (단어 2회 초과 반복)
    processed_text = re.sub(r'(\b\w+\b)([\s,.]+\1){' + str(max_repeat_times -1) + r',}', r'\1', text)
    
    # 짧은 구문 반복 (예: "알겠습니다 알겠습니다")
    # 이 부분은 더 정교한 방법 필요 (예: N-gram 분석)
    # 현재는 위 정규식이 어느 정도 커버할 수 있다고 가정
    return processed_text


def _post_process_transcription(raw_text: str) -> str:
    """
    STT 결과를 후처리하여 가독성을 높이고 오류를 줄입니다.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # 1. 기본적인 공백 및 구두점 정리
    text = _normalize_whitespace_and_punctuation(raw_text)

    # 2. 기본적인 반복 제거
    text = _remove_basic_repetitions(text, max_repeat_times=2) # 2번 초과 반복(즉, 3번 이상)을 줄임

    # (선택적) 추가적인 후처리 로직:
    # - 문맥에 맞지 않는 짧은 단어 제거 (예: "어", "음" 등 필러 단어 제거 - 단, 의도된 발화일 수 있으므로 주의)
    # - 문장 경계 복원 (만약 Whisper가 문장 구분을 잘 못했을 경우)
    # - 사용자 정의 단어 교정 (자주 오인식되는 단어 목록 기반)

    # 기존 stt.py의 remove_duplicates 함수 (문장 단위 중복/포함 제거) 로직 적용 여부 결정
    # 이 로직은 때때로 너무 많은 내용을 제거할 수 있으므로, 신중하게 적용하거나 파라미터화 필요
    # text = _apply_stt_py_sentence_deduplication(text)

    return text

# (선택적) 기존 stt.py의 문장 단위 중복 제거 로직 (필요시 _post_process_transcription 내부에서 호출)
def _apply_stt_py_sentence_deduplication(text: str) -> str:
    # 기존 stt.py의 remove_duplicates 함수 로직 (약간 수정)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences: return ""

    processed_sentences = []
    last_added_sentence_lower = ""

    for sentence_orig in sentences:
        sentence_strip = sentence_orig.strip()
        if not sentence_strip: continue

        current_sentence_lower = sentence_strip.lower()

        if not processed_sentences: # 첫 문장
            processed_sentences.append(sentence_strip)
            last_added_sentence_lower = current_sentence_lower
            continue

        # 완전 중복
        if current_sentence_lower == last_added_sentence_lower:
            continue

        # 부분 중복 (포함 관계) - 기존 stt.py 로직 (길이 10자 이상 조건은 제외하고 테스트)
        # 이 로직은 때로 과도할 수 있음.
        # if last_added_sentence_lower in current_sentence_lower and len(last_added_sentence_lower) > 5: # 이전이 현재에 포함
        #     processed_sentences[-1] = sentence_strip # 현재 것으로 대체 (더 긴 정보 가정)
        #     last_added_sentence_lower = current_sentence_lower
        #     continue
        # elif current_sentence_lower in last_added_sentence_lower and len(current_sentence_lower) > 5: # 현재가 이전에 포함
        #     continue # 현재 것 추가 안 함

        processed_sentences.append(sentence_strip)
        last_added_sentence_lower = current_sentence_lower
        
    return " ".join(processed_sentences)


# === STT 핵심 처리 함수 (Hugging Face Pipeline 활용) ===
async def _perform_stt_with_pipeline(
    audio_path: str,
    stt_pipeline: Any, # 미리 로드된 Hugging Face pipeline 객체
    language: Optional[str] = "ko", # 대상 언어 (pipeline 생성 시 지정 가능)
    chunk_length_s: int = 30,       # 파이프라인 청크 길이 (초)
    stride_length_s: int = 5,       # 파이프라인 스트라이드 길이 (초, 오버랩 관련)
    # generate_kwargs: Optional[dict] = None # 모델 생성 관련 추가 파라미터
) -> str:
    """
    Hugging Face ASR Pipeline을 사용하여 오디오 파일에서 텍스트를 추출합니다.
    """
    if not _PIPELINE_AVAILABLE or stt_pipeline is None:
        print("STT 서비스 경고: ASR 파이프라인 사용 불가. 더미 결과를 반환합니다.")
        await asyncio.sleep(0.1) # 비동기 함수 흉내
        return f"[더미 STT(Pipeline): {os.path.basename(audio_path)} (파이프라인/패키지 문제)]"

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"STT 서비스 내부 오류: 오디오 파일 없음 - {audio_path}")

    print(f"  Pipeline STT 처리 시작: {audio_path}, lang={language}, chunk={chunk_length_s}s, stride={stride_length_s}s")
    
    try:
        # Hugging Face pipeline은 파일 경로를 직접 받거나, 로드된 오디오 데이터를 받을 수 있음
        # 여기서는 파일 경로를 사용
        # generate_kwargs = generate_kwargs or {} # whisper 모델의 generate 함수에 전달될 인자들
        # if language and "language" not in generate_kwargs: # 파이프라인 생성시 언어 지정했다면 불필요할 수 있음
        #     generate_kwargs["language"] = language
        
        # 파이프라인 호출 시 generate_kwargs 전달 방식 확인 필요
        # (예시: result = stt_pipeline(audio_path, generate_kwargs=generate_kwargs, ...))

        # 현재 Whisper 파이프라인은 generate_kwargs를 직접 받지 않고,
        # pipeline 생성 시 model_kwargs에 전달하거나,
        # 또는 내부적으로 최적화된 값을 사용할 수 있음.
        # 언어 설정 등은 pipeline 생성 시 또는 호출 시 파라미터로 전달 가능
        whisper_params = {}
        if language:
            whisper_params["language"] = language # whisper pipeline이 이 파라미터를 지원하는지 확인 필요

        # STT 파이프라인 실행
        # 로컬 파일의 경우, 파이프라인이 내부적으로 파일을 읽음.
        # 매우 큰 파일의 경우, 메모리 문제를 피하기 위해 스트리밍 방식으로 읽도록 하거나,
        # 파일을 직접 파이프라인에 전달하기 전에 바이트로 읽어서 전달할 수 있음.
        # 예: with open(audio_path, "rb") as f: audio_bytes = f.read() -> 파이프라인(audio_bytes)
        # 이 부분은 pipeline 구현에 따라 달라짐. 기본적으로는 파일 경로 전달.

        transcription_result = stt_pipeline(
            audio_path,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            return_timestamps=False, # 타임스탬프 필요시 True로 하고 후처리
            # language=language, # pipeline 생성 시 지정했다면 중복일 수 있음
            # 아래는 whisper 모델에 직접 전달하고 싶을 때 고려 (pipeline 생성 시 model_kwargs 사용)
            # generate_kwargs={"temperature": 0.1, "no_repeat_ngram_size": 3}
        )
        
        raw_text = transcription_result["text"] if isinstance(transcription_result, dict) and "text" in transcription_result else str(transcription_result)

    except Exception as e:
        print(f"STT 서비스 오류: Pipeline 처리 중 예외 - {e}")
        # 파이프라인 오류 시 더 상세한 정보 로깅 필요
        raise RuntimeError(f"ASR Pipeline 처리 중 오류 발생: {e}") from e
    
    # 후처리 적용
    final_text = _post_process_transcription(raw_text)
    print(f"  Pipeline STT 처리 완료 (후처리 전 길이: {len(raw_text)}, 후 길이: {len(final_text)})")
    return final_text


# === FastAPI 서비스 인터페이스 함수 (라우터에서 호출됨) ===
async def process_uploaded_rc_file_to_text(
    rc_file: UploadFile,
    stt_pipeline_instance: Any, # app.state 등에서 주입받을 미리 로드된 ASR 파이프라인
    # API를 통해 받고 싶은 STT 관련 설정 파라미터들:
    target_language: Optional[str] = "ko",
    pipeline_chunk_length_s: int = 30, # 기본값 30초
    pipeline_stride_length_s: int = 5   # 기본값 5초 (오버랩)
) -> str:
    """
    업로드된 녹음 파일(rc_file: UploadFile)을 받아 STT 처리를 수행하고 텍스트를 반환합니다.
    Hugging Face ASR Pipeline을 사용합니다.
    """
    print(f"STT 서비스 수신: '{rc_file.filename}' (타입: {rc_file.content_type}), 언어: {target_language}")
    service_start_time = time.time()

    unique_id = uuid.uuid4()
    file_extension = os.path.splitext(rc_file.filename)[1] if rc_file.filename else ".m4a" # 기본 확장자
    temp_file_name = f"upload_{unique_id}{file_extension}"
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_file_name)

    try:
        # 1. UploadFile을 임시 파일로 저장
        # (참고: ffmpeg_read는 바이트 배열을 직접 받을 수 있으므로, 파일 저장 없이 메모리에서 처리 가능성도 있음)
        # 하지만 파이프라인이 파일 경로를 더 잘 처리하는 경우가 많고, 디버깅에도 용이.
        file_content = await rc_file.read() # 파일을 메모리로 읽음 (큰 파일 주의)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)
        
        print(f"  임시 파일 저장: {temp_file_path} (크기: {len(file_content)} bytes)")

        # (선택적) 오디오 파일 유효성 검사 또는 기본 정보 로깅
        # if _PACKAGES_AVAILABLE:
        #     try:
        #         audio_segment = AudioSegment.from_file(temp_file_path, format=file_extension.lstrip('.'))
        #         print(f"  오디오 정보: 길이={audio_segment.duration_seconds():.2f}s, 채널={audio_segment.channels}, 프레임률={audio_segment.frame_rate}")
        #     except Exception as e_audio_info:
        #         print(f"STT 서비스 경고: 업로드된 오디오 파일 정보 읽기 실패 - {e_audio_info}")


        # 2. STT 핵심 처리 함수 호출 (동기 함수이므로 비동기적으로 실행)
        transcribed_text = await asyncio.to_thread(
            _perform_stt_with_pipeline,
            audio_path=temp_file_path,
            stt_pipeline=stt_pipeline_instance,
            language=target_language,
            chunk_length_s=pipeline_chunk_length_s,
            stride_length_s=pipeline_stride_length_s
        )
        
        processing_time = time.time() - service_start_time
        print(f"STT 서비스 완료: '{rc_file.filename}'. 소요시간: {_format_time(processing_time)}")
        return transcribed_text

    except FileNotFoundError as e_fnf:
        print(f"STT 서비스 오류 (파일): '{rc_file.filename}' - {e_fnf}")
        raise RuntimeError(f"처리할 파일을 찾을 수 없습니다: {rc_file.filename}") from e_fnf
    except ValueError as e_val: # 오디오 포맷 문제 등
        print(f"STT 서비스 오류 (데이터): '{rc_file.filename}' - {e_val}")
        raise ValueError(f"제공된 파일의 데이터 형식이 올바르지 않거나 처리할 수 없습니다: {rc_file.filename}") from e_val
    except RuntimeError as e_rt: # 모델 실행 문제 등
        print(f"STT 서비스 오류 (런타임): '{rc_file.filename}' - {e_rt}")
        raise RuntimeError(f"음성 인식 처리 중 내부 서버 오류가 발생했습니다.") from e_rt
    except Exception as e:
        print(f"STT 서비스 알 수 없는 오류 ({rc_file.filename}): {type(e).__name__} - {e}")
        raise RuntimeError(f"음성 인식 처리 중 알 수 없는 오류가 발생했습니다.") from e
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_remove:
                print(f"STT 서비스 경고: 임시 파일 삭제 실패 '{temp_file_path}' - {e_remove}")
        
        if not rc_file.file.closed: # type: ignore
            try:
                await rc_file.close()
            except Exception as e_close: # pragma: no cover
                print(f"STT 서비스 경고: 업로드 파일 닫기 실패 '{rc_file.filename}' - {e_close}")