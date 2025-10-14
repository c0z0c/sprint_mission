import logging
import pytz
from datetime import datetime

class ShortLevelFormatter(logging.Formatter):
    """로그 레벨을 1글자로 축약 (DEBUG→D, INFO→I, WARNING→W, ERROR→E, CRITICAL→C)"""

    LEVEL_MAP = {
        'DEBUG': 'D',
        'INFO': 'I',
        'WARNING': 'W',
        'ERROR': 'E',
        'CRITICAL': 'C'
    }
    kst = pytz.timezone('Asia/Seoul')

    def format(self, record):
        # 원본 레벨명을 약자로 교체
        record.levelname = self.LEVEL_MAP.get(record.levelname, record.levelname)
        return super().format(record)

    def formatTime(self, record, datefmt=None):
        """record.created를 KST로 변환해 포맷된 문자열 반환"""
        ct = datetime.fromtimestamp(record.created, tz=self.kst)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime('%Y-%m-%d %H:%M:%S')
    
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# 기존 핸들러의 Formatter 교체
for handler in logging.getLogger().handlers:
    handler.setFormatter(ShortLevelFormatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))