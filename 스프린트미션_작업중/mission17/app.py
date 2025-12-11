# -*- coding: utf-8 -*-
"""Streamlit ONNX MNIST 숫자 예측 서비스 - 실제 ONNX 모델 통합"""

import datetime
import hashlib
import logging
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from helper_dev_utils import get_auto_logger

logger = get_auto_logger()

# Import src modules
from src.model import MNISTPipeline, PredictionResult
from src.history import HistoryManager, HistoryRecord, FileHistoryManager
from src.visualization import VisualizationManager


@st.cache_resource
def load_mnist_pipeline() -> MNISTPipeline:
    """MNIST 파이프라인을 로드하고 캐싱합니다.

    @st.cache_resource 데코레이터를 사용하여 모델을 한 번만 로드하고
    세션 간에 재사용합니다.

    Returns:
        초기화된 MNISTPipeline 객체
    """
    pipeline = MNISTPipeline()
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
    """matplotlib 한글 폰트 설정"""
    from helper_plot_hangul import matplotlib_font_reset


def initialize_session_state() -> None:
    """세션 상태 변수 초기화"""
    if "history_manager" not in st.session_state:
        # st.session_state.history_manager = HistoryManager(max_records=100)
        st.session_state.history_manager = FileHistoryManager(
            save_dir="./history", max_records=100, auto_save=True
        )
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0


def display_history() -> None:
    """예측 히스토리를 테이블 형식으로 표시"""
    history_manager = st.session_state.history_manager
    records = history_manager.get_all_records(reverse=True)

    if not records:
        st.info("아직 예측 기록이 없습니다")
        return

    st.markdown("### 예측 기록")

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
            st.divider()


def main():
    """메인 애플리케이션"""

    # 페이지 설정
    st.set_page_config(
        page_title="AI 숫자 예측",
        page_icon="🔢",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # matplotlib 한글 폰트 설정
    setup_matplotlib_font()

    # 세션 상태 초기화
    initialize_session_state()

    # 모델 및 시각화 매니저 로드
    pipeline = load_mnist_pipeline()
    viz_manager = load_visualization_manager()
    history_manager = st.session_state.history_manager

    # 제목
    st.subheader("🔢 AI 숫자 예측")
    st.markdown("---")

    # 메인 레이아웃 (2열)
    col1, col2, col3 = st.columns([1, 1, 1])

    # 좌측: 캔버스 영역
    with col1:
        st.markdown("### 입력 캔버스")
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
        st.markdown("### 전처리 이미지")
        preprocessed_placeholder = st.empty()

    with col3:
        # 추론 결과 영역
        st.markdown("### 추론 결과")
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
    st.markdown("---")
    st.markdown("#### 📚 이미지 저장소")

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

    display_history()


if __name__ == "__main__":
    main()
