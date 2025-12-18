"""
리뷰 게시판 페이지 (board.py)
board_edit.py와 board_list.py를 통합하는 메인 페이지
"""

import sys
from pathlib import Path

# pages 디렉토리를 sys.path에 추가
pages_dir = Path(__file__).parent
if str(pages_dir) not in sys.path:
    sys.path.insert(0, str(pages_dir))

import streamlit as st
from board_edit import ReviewEditManager
from board_list import ReviewListManager


class ReviewManager:
    """
    리뷰 관리 통합 클래스
    """

    def __init__(self):
        """ReviewManager 초기화"""
        self.edit_manager = ReviewEditManager()
        self.list_manager = ReviewListManager()

    def render(self):
        """리뷰 게시판 페이지 렌더링"""
        st.write("##### 리뷰 게시판")
        st.write("영화 리뷰를 작성하고 AI 감성 분석 결과를 확인할 수 있습니다.")

        # 탭 구성
        tab1, tab2 = st.tabs(["리뷰 작성", "리뷰 목록"])

        with tab1:
            self.edit_manager.render()

        with tab2:
            self.list_manager.render()


# 페이지 실행
if __name__ == "__main__":
    manager = ReviewManager()
    manager.render()
