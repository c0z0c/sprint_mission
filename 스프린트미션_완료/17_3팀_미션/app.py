# -*- coding: utf-8 -*-
"""Streamlit ONNX MNIST 숫자 예측 서비스 - 실제 ONNX 모델 통합

이 모듈은 Streamlit을 사용하여 손으로 그린 숫자를 AI가 예측하는 웹 애플리케이션입니다.

주요 기능:
    - 캔버스에 0-9 숫자 그리기
    - ONNX 모델을 사용한 실시간 숫자 예측
    - 예측 결과 시각화 (확률 분포 차트)
    - 예측 기록 저장 및 관리
    - 모델 URL 동적 변경 및 다운로드
    - 이미지 해시 기반 중복 예측 방지

기술 스택:
    - Streamlit: 웹 UI 프레임워크
    - ONNX Runtime: AI 모델 추론
    - OpenCV/PIL: 이미지 전처리
    - Matplotlib: 결과 시각화
"""

import datetime
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import streamlit as st
from helper_dev_utils import get_auto_logger
from helper_plot_hangul import matplotlib_font_reset
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.history import FileHistoryManager, HistoryManager, HistoryRecord

# Import src modules
from src.model import MNISTPipeline, ModelConfig, ModelDownloader, PredictionResult
from src.utils.utils_st import (
    hidden_page_top_margin,
    minimal_divider,
)
from src.visualization import VisualizationManager

logger = get_auto_logger(log_level=logging.DEBUG)


@st.cache_resource
def load_mnist_pipeline(_config) -> MNISTPipeline:
    """MNIST 파이프라인을 로드하고 캐싱합니다.

    @st.cache_resource 데코레이터를 사용하여 모델을 한 번만 로드하고
    세션 간에 재사용합니다. 언더스코어 접두사는 Streamlit이 파라미터를 해싱하지 않도록 합니다.

    Args:
        _config: ModelConfig 객체 (언더스코어 접두사로 해싱 방지)

    Returns:
        초기화된 MNISTPipeline 객체
    """
    pipeline = MNISTPipeline(config=_config)
    pipeline.initialize()
    return pipeline


@st.cache_resource
def load_visualization_manager() -> VisualizationManager:
    """시각화 매니저를 로드하고 캐싱합니다.

    Returns:
        VisualizationManager 객체
    """
    return VisualizationManager()


def setup_matplotlib_font() -> None:
    """matplotlib 한글 폰트 설정

    한글이 깨지지 않도록 matplotlib의 폰트를 설정합니다.
    현재는 helper_plot_hangul 모듈에서 자동으로 처리되므로 별도 설정이 불필요합니다.
    """
    pass


def initialize_session_state() -> None:
    """세션 상태 변수 초기화

    Streamlit 세션 상태에 필요한 변수들을 초기화합니다.
    - history_manager: 예측 기록을 파일 시스템에 저장하는 매니저
    - canvas_key: 캔버스 위젯의 고유 키 (캔버스 초기화용)
    - model_config: ONNX 모델 설정 정보
    """
    if "history_manager" not in st.session_state:
        # st.session_state.history_manager = HistoryManager(max_records=100)
        st.session_state.history_manager = FileHistoryManager(
            save_dir="./history", max_records=100, auto_save=True
        )
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "model_config" not in st.session_state:
        st.session_state.model_config = ModelConfig()


def extract_model_name(url: str) -> str:
    """URL에서 파일명 추출

    Args:
        url: ONNX 모델 다운로드 URL

    Returns:
        추출된 파일명 (기본값: model.onnx)
    """

    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename or not filename.endswith(".onnx"):
        return "model.onnx"
    return filename


def validate_model_url(url: str) -> Optional[str]:
    """모델 URL 형식 검증

    Args:
        url: 검증할 URL

    Returns:
        에러 메시지 (문제가 있을 경우) 또는 None (정상인 경우)
    """
    if not url:
        return "URL을 입력해주세요"
    if not url.startswith(("http://", "https://")):
        return "URL은 http:// 또는 https://로 시작해야 합니다"
    if not url.endswith(".onnx"):
        return "URL은 .onnx 파일을 가리켜야 합니다"
    return None


def validate_mnist_model(model_path, config) -> Optional[str]:
    """ONNX 모델 MNIST 호환성 검증

    다운로드된 ONNX 모델이 MNIST 숫자 예측에 사용 가능한지 검증합니다.
    입력 shape(1, 1, 28, 28)과 출력 클래스 수(10)를 확인합니다.

    Args:
        model_path: ONNX 모델 파일 경로
        config: 모델 설정 객체 (입력 shape 및 클래스 수 정보)

    Returns:
        에러 메시지 (호환되지 않을 경우) 또는 None (호환될 경우)
    """
    try:

        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        # 입력 shape 확인
        input_shape = session.get_inputs()[0].shape
        expected_shape = list(config.input_shape)

        # 동적 배치 차원 허용
        if input_shape[0] in [None, "N", "batch"]:
            input_shape = [1] + list(input_shape[1:])

        # 공간 차원 (28x28) 및 채널 (1) 확인
        if list(input_shape[1:]) != expected_shape[1:]:
            return f"입력 shape 불일치: 예상 {expected_shape}, 실제 {input_shape}"

        # 출력 클래스 수 확인
        output_shape = session.get_outputs()[0].shape
        if output_shape[-1] != config.num_classes:
            return f"출력 클래스 수 불일치: 예상 {config.num_classes}, 실제 {output_shape[-1]}"

        return None

    except Exception as e:
        return f"모델 로드 실패: {str(e)}"


def display_model_settings() -> None:
    """사이드바에 모델 URL 설정 UI 표시

    사용자가 ONNX 모델 URL을 입력하고 모델을 다운로드할 수 있는 UI를 제공합니다.
    - URL 입력 및 파일명 자동 추출
    - 모델 다운로드 및 검증
    - 기본값으로 초기화 기능
    """

    with st.expander("⚙️ 모델 설정", expanded=False):

        st.markdown("##### ONNX 모델 구성")

        # 현재 모델 정보
        current_config = st.session_state.model_config
        st.info(f"현재 모델: {current_config.model_name}")

        extracted_name = current_config.model_name

        cols = st.columns([8, 2])
        with cols[0]:
            # URL 입력
            model_url = st.text_input(
                "모델 URL",
                value=current_config.model_url,
                help="ONNX 모델 URL을 입력하세요 (.onnx로 끝나야 함)",
                key="model_url_input",
            )
            if model_url != current_config.model_url:
                extracted_name = extract_model_name(model_url)
            else:
                extracted_name = current_config.model_name

        with cols[1]:
            # 모델명 자동 추출
            model_name = st.text_input(
                "모델 파일명",
                value=extracted_name,
                help="캐시될 모델 파일명",
                key="model_name_input",
            )

        cols = st.columns([1, 1])
        with cols[0]:
            # 적용 버튼
            if st.button(
                "적용 및 다운로드", key="apply_model_btn", use_container_width=True
            ):
                logger.debug(f"모델 설정 적용: URL={model_url}, 이름={model_name}")

                # URL 검증
                error = validate_model_url(model_url)
                if error:
                    st.error(error)
                    return

                # 모델명 검증
                if not model_name.endswith(".onnx"):
                    st.error("모델 파일명은 .onnx로 끝나야 합니다")
                    return

                # 새 설정 생성
                new_config = ModelConfig(
                    model_url=model_url,
                    model_name=model_name,
                    cache_dir="./models",
                    input_shape=(1, 1, 28, 28),
                    num_classes=10,
                )

                # 다운로드 및 검증
                try:
                    with st.spinner("모델 다운로드 중..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(progress: float, status: str = ""):
                            progress_bar.progress(min(progress, 1.0))
                            if status:
                                status_text.text(status)

                        # 모델 다운로드
                        downloader = ModelDownloader(config=new_config)
                        model_path = downloader.download(
                            force=True, progress_callback=update_progress
                        )

                        # 모델 호환성 검증
                        validation_error = validate_mnist_model(model_path, new_config)
                        if validation_error:
                            st.error(f"모델 검증 실패: {validation_error}")
                            # 잘못된 모델 삭제
                            model_path.unlink(missing_ok=True)
                            return

                        # 세션 상태 업데이트
                        st.session_state.model_config = new_config

                        # 캐시 초기화 (새 모델 로드)
                        st.cache_resource.clear()

                        st.success("✅ 모델 다운로드 및 검증 완료!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ 다운로드 실패: {str(e)}")
                    logger.exception("모델 다운로드 오류")

        with cols[1]:
            # 초기화 버튼
            if st.button(
                "기본값으로 초기화", key="reset_model_btn", use_container_width=True
            ):
                st.session_state.model_config = ModelConfig()
                st.cache_resource.clear()
                st.success("기본 모델로 초기화되었습니다")
                st.rerun()


def display_history() -> None:
    """예측 히스토리를 테이블 형식으로 표시

    저장된 모든 예측 기록을 최신순으로 표시합니다.
    - 썸네일 이미지
    - 예측된 숫자 및 신뢰도
    - 예측 시각
    - 선택적으로 확률 분포 차트 표시
    """
    history_manager = st.session_state.history_manager
    records = history_manager.get_all_records(reverse=True)

    if not records:
        st.info("아직 예측 기록이 없습니다")
        return

    st.markdown("###### 예측 기록")

    show_charts = st.checkbox(
        "확률 분포 시각화 표시",
        value=False,
        help="각 예측의 전체 확률 분포를 작은 차트로 표시합니다",
    )

    # 최신 항목부터 표시 (역순)
    for idx, record in enumerate(records):
        with st.container():
            if show_charts:
                cols = st.columns([1, 2, 2, 2, 3, 3])  # 차트용 6번째 열
            else:
                cols = st.columns([1, 2, 2, 2, 3])  # 원래 5열

            with cols[0]:
                st.write(f"**#{record.record_id}**")

            with cols[1]:
                # 썸네일 이미지 표시 (원본 캔버스 이미지)
                thumbnail = Image.fromarray(record.canvas_image.astype("uint8"))
                st.image(thumbnail, width=60)

            with cols[2]:
                st.write(f"**예측값:** {record.predicted_label}")

            with cols[3]:
                st.write(f"**신뢰도:** {record.confidence:.2%}")

            with cols[4]:
                st.write(f"**시각:** {record.timestamp}")

            if show_charts:
                with cols[5]:
                    # 시각화 매니저 로드
                    viz_manager = load_visualization_manager()

                    # 작은 차트 생성
                    fig = viz_manager.prediction_viz.plot_compact_bar_chart(
                        record.probabilities, record.predicted_label
                    )

                    # 차트 표시
                    st.pyplot(fig, width="stretch")

                    # 메모리 누수 방지를 위한 정리
                    plt.close(fig)

        if idx < len(records) - 1:
            minimal_divider()
            # st.divider()


def main():
    """메인 애플리케이션

    Streamlit 앱의 진입점입니다. 다음과 같은 순서로 실행됩니다:
    1. 페이지 설정 및 초기화
    2. 모델 및 시각화 매니저 로드
    3. UI 렌더링 (캔버스, 설정, 결과 표시)
    4. 예측 로직 처리 (버튼 클릭 시)
    5. 히스토리 표시
    """

    # 페이지 설정
    st.set_page_config(
        page_title="AI 숫자 예측",
        page_icon="🔢",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    hidden_page_top_margin()

    # matplotlib 한글 폰트 설정
    setup_matplotlib_font()

    # 세션 상태 초기화
    initialize_session_state()

    # 모델 및 시각화 매니저 로드 (세션 상태의 config 사용)
    pipeline = load_mnist_pipeline(st.session_state.model_config)
    viz_manager = load_visualization_manager()
    history_manager = st.session_state.history_manager

    # 모델 설정 UI 추가
    display_model_settings()

    # 제목
    st.markdown("##### 🔢 AI 숫자 예측")
    st.caption(f"현재 적용 모델: {st.session_state.model_config.model_name}")

    minimal_divider()

    # 메인 레이아웃 (2열)
    col1, col2, col3 = st.columns([1, 1, 1])

    # 좌측: 캔버스 영역
    with col1:
        st.markdown("###### 입력 캔버스")
        # st.write("아래 캔버스에 0-9 사이의 숫자를 그려주세요")

        left, center, right = st.columns([1, 4, 2])
        with left:
            st.write("0 - 9<br/>사이의<br/>숫자를<br/>그리기", unsafe_allow_html=True)

        with right:
            use_bbox_resize = st.checkbox(
                "전처리",
                value=True,
                help="체크 시, 그려진 숫자의 바운딩 박스를 추출하여 비율을 유지하며 리사이즈합니다. "
                "체크 해제 시, 전체 캔버스를 28x28로 직접 리사이즈합니다.",
            )

        with center:
            canvas_result = st_canvas(
                stroke_width=5,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=200,
                height=200,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state.canvas_key}",
                display_toolbar=False,  # True: 툴바 표시 (기본값), False: 툴바 숨김
            )

        # 캔버스 바로 아래에 버튼을 가로로 배치 (두 버튼을 가운데에 유지)
        btn_left, btn_right = st.columns([1, 1])
        with btn_left:
            predict_button = st.button("예측하기", use_container_width=True)
        with btn_right:
            if st.button("캔버스 지우기", use_container_width=True):
                st.session_state.canvas_key += 1
                st.rerun()

    # 우측: 전처리 이미지 및 추론 결과 영역
    with col2:
        # 전처리 이미지 영역
        st.markdown("###### 전처리 이미지")
        preprocessed_placeholder = st.empty()

    with col3:
        # 추론 결과 영역
        st.markdown("###### 추론 결과")
        result_placeholder = st.empty()

    # 예측 버튼 클릭 로직
    if predict_button:
        if canvas_result.image_data is not None:
            # 캔버스가 비어있는지 확인 (모든 픽셀이 흰색인지)
            canvas_image = canvas_result.image_data.astype(np.uint8)
            if np.all(canvas_image[:, :, 3] == 0):  # 알파 채널 확인
                st.warning("캔버스에 숫자를 그려주세요!")
            else:
                with st.spinner("예측 중..."):
                    # 1단계: 원본 이미지로 해시 계산
                    image_hash = HistoryRecord.compute_image_hash(
                        canvas_image, use_bbox_resize
                    )

                    # 2단계: 히스토리에서 동일 해시 검색
                    existing_record = history_manager.find_by_hash(image_hash)

                    if existing_record is not None:
                        # 기존 예측 결과 재사용
                        logger.debug(
                            f"동일 이미지 발견 (해시: {image_hash[:16]}...), 기존 결과 재사용"
                        )
                        prediction_result = PredictionResult(
                            predicted_label=existing_record.predicted_label,
                            confidence=existing_record.confidence,
                            probabilities=existing_record.probabilities,
                            preprocessed_image=existing_record.preprocessed_image,
                        )
                    else:
                        # 새로운 이미지, 모델 추론 수행 (전처리 포함)
                        logger.debug(
                            f"새로운 이미지 (해시: {image_hash[:16]}...), 모델 추론 수행"
                        )
                        prediction_result = pipeline.predict(
                            canvas_image, use_bbox_resize
                        )

                # 전처리 이미지 표시
                with preprocessed_placeholder.container():
                    if prediction_result.preprocessed_image is not None:
                        st.image(
                            prediction_result.preprocessed_image,
                            caption="전처리 28x28 (반전 및 정규화)",
                            width=200,
                        )

                # 추론 결과 표시
                with result_placeholder.container():
                    predicted_html = "<h4 style='text-align: center;'>"
                    predicted_html += f"<span style='color: #ff6b6b;'>예측 숫자: [{prediction_result.predicted_label}]</span>"
                    predicted_html += f"<span style='color: #000000;'>신뢰도: {prediction_result.confidence:.2%}</span></h4>"
                    predicted_html += "</h4>"
                    st.markdown(predicted_html, unsafe_allow_html=True)

                    # VisualizationManager를 사용한 막대 차트
                    fig = viz_manager.prediction_viz.plot_bar_chart(
                        prediction_result.probabilities,
                        prediction_result.predicted_label,
                        title="예측 확률 분포",
                    )
                    st.pyplot(fig)

                # 히스토리에 추가 (새로운 이미지인 경우만)
                if existing_record is not None:
                    st.success("기존 예측 결과를 재사용했습니다!")
                else:
                    history_manager.add_record(
                        canvas_image=canvas_image,
                        preprocessed_image=prediction_result.preprocessed_image,
                        predicted_label=prediction_result.predicted_label,
                        confidence=prediction_result.confidence,
                        probabilities=prediction_result.probabilities,
                        image_hash=image_hash,
                        notes=None,
                    )
                    st.success("예측이 완료되었습니다!")

        else:
            st.warning("캔버스에 숫자를 그려주세요!")

    else:
        # 초기 상태 메시지
        with preprocessed_placeholder.container():
            st.info("캔버스에 숫자를 그리고 예측 버튼을 클릭하세요")

        with result_placeholder.container():
            st.info("예측 결과가 여기 표시됩니다")

    # 하단: 이미지 저장소 (히스토리)
    # st.markdown("---")
    minimal_divider()
    st.markdown("###### 📚 이미지 저장소")

    if len(history_manager) > 0:
        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            if st.button("히스토리 전체 삭제"):
                history_manager.clear_all()
                st.rerun()

        with col_btn2:
            stats = history_manager.get_statistics()
            st.write(
                f"**총 {stats['total_count']}개 기록** | 평균 신뢰도: {stats['avg_confidence']:.2%}"
            )

        minimal_divider()

    display_history()


if __name__ == "__main__":
    main()
