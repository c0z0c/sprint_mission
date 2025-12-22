"""
검색 UI 공통 모듈
영화 검색 및 리뷰 검색 UI 컴포넌트
"""

import streamlit as st
from typing import Dict, Optional, Tuple
import logging
from helper_dev_utils import get_auto_logger

from .MovieSearchUI import MovieSearchUI
from .ReviewSearchUI import ReviewSearchUI

logger = get_auto_logger(log_level=logging.DEBUG)
