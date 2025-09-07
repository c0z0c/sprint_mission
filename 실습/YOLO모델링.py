# pip install PyQt5 matplotlib ultralytics opencv-python

from urllib.request import urlretrieve; urlretrieve("https://raw.githubusercontent.com/c0z0c/jupyter_hangul/refs/heads/beta/helper_c0z0c_dev.py", "helper_c0z0c_dev.py")
import importlib
import helper_c0z0c_dev as helper
importlib.reload(helper)

import cv2
import time
import sys
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QLabel, QPushButton, QTextEdit, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QMutex, QWaitCondition
from PyQt5.QtGui import QFont, QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import queue
import threading

# 한글 UI 텍스트 딕셔너리 선언
lang_kr = {}

class CameraThread(QThread):
    """카메라 영상 캡처를 위한 스레드 (고속)"""
    frame_signal = pyqtSignal(np.ndarray)  # 원본 프레임 전송
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = None
        
    def run(self):
        # 카메라 초기화 (DirectShow 사용하여 윈도우에서 최적화)
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return
        
        # 카메라 설정: 해상도, FPS, 버퍼 크기 최적화
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 30 FPS로 설정
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기를 1로 설정하여 지연 최소화
        
        print("카메라 초기화 완료")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    # 프레임을 GUI로 전송 (복사본 전송으로 메모리 안전성 확보)
                    self.frame_signal.emit(frame.copy())
                else:
                    print("프레임을 읽을 수 없습니다.")
                    time.sleep(0.1)  # 프레임 읽기 실패 시 잠시 대기
                    continue
                    
                self.msleep(33)  # 약 30 FPS (1000ms/30 = 33ms)
        except Exception as e:
            print(f"Camera error: {e}")
        finally:
            if self.cap:
                self.cap.release()
                print("카메라 해제됨")
    
    def stop(self):
        self.running = False

class YOLODetectionThread(QThread):
    """YOLO 탐지를 위한 별도 스레드 (0.2초마다 처리)"""
    detection_signal = pyqtSignal(dict)  # 탐지 결과 전송
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)  # 최대 1개 프레임만 저장 (최신 프레임만)
        self.class_names = ['cat', 'dog']
        self.detection_counts = {'cat': 0, 'dog': 0, 'none': 0}
        self.frame_count = 0
        self.start_time = time.time()
        self.last_process_time = 0
        
    def add_frame(self, frame):
        """새 프레임 추가 (0.2초마다만 처리하여 성능 최적화)"""
        current_time = time.time()
        
        # 0.2초(200ms)가 지났는지 확인
        if current_time - self.last_process_time >= 0.2:
            try:
                # 큐에 기존 프레임이 있으면 제거하고 새 프레임 추가 (최신 프레임만 유지)
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put_nowait(frame)
                self.last_process_time = current_time
            except queue.Full:
                pass  # 큐가 가득 찬 경우 무시
    
    def run(self):
        # YOLO 모델 로드
        model = YOLO(self.model_path)
        print("YOLO model loaded successfully")
        
        try:
            while self.running:
                try:
                    # 큐에서 프레임 가져오기 (0.3초 대기)
                    frame = self.frame_queue.get(timeout=0.3)
                    
                    # YOLO 탐지 수행 (confidence 임계값 0.6)
                    results = model(frame, verbose=False, conf=0.6)
                    current_time = time.time() - self.start_time
                    
                    # YOLO 결과를 프레임에 그리기 (바운딩 박스, 라벨 표시)
                    annotated_frame = results[0].plot()
                    
                    # 탐지 결과 처리 및 카운팅
                    current_detection = "No Detection"
                    if len(results[0].boxes) > 0:
                        detected_objects = []
                        for box in results[0].boxes:
                            cls = int(box.cls.item())
                            confidence = float(box.conf.item())
                            detected_class = self.class_names[cls]
                            self.detection_counts[detected_class] += 1
                            detected_objects.append(f"{detected_class}({confidence:.2f})")
                        
                        current_detection = ", ".join(detected_objects)
                    else:
                        self.detection_counts['none'] += 1
                    
                    self.frame_count += 1
                    
                    # GUI로 탐지 결과 전송
                    data = {
                        'frame_count': self.frame_count,
                        'elapsed_time': current_time,
                        'detection_counts': self.detection_counts.copy(),
                        'current_detection': current_detection,
                        'annotated_frame': annotated_frame
                    }
                    self.detection_signal.emit(data)
                    
                    print(f"YOLO processed frame {self.frame_count}: {current_detection}")
                    
                except queue.Empty:
                    continue  # 프레임이 없으면 계속 대기
                except Exception as e:
                    print(f"YOLO detection error: {e}")
                    break
                    
        except Exception as e:
            print(f"YOLO thread error: {e}")
    
    def stop(self):
        self.running = False

class StatisticsCanvas(FigureCanvas):
    """통계 차트를 위한 matplotlib 캔버스"""
    def __init__(self, parent=None):
        # Figure 생성 방식 수정 - matplotlib 호환성 문제 해결
        self.fig = Figure()
        self.fig.set_size_inches(8, 6)
        self.fig.set_dpi(100)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # 서브플롯 생성 (2x2 그리드)
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # 막대 그래프
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # 파이 차트
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # 최종 결과
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # 성능 정보
        
        # 차트 제목 설정
        try:
            chart_title = lang_kr.get('chart_title', 'YOLO 탐지 통계')
            self.fig.suptitle(chart_title, fontsize=20, weight='bold')
            self.fig.tight_layout(pad=2.0)
        except Exception as e:
            print(f"Chart title setting error: {e}")
            # 기본 설정으로 fallback
            self.fig.suptitle('YOLO Detection Statistics', fontsize=16)
            self.fig.tight_layout()
        
    def update_charts(self, data):
        """실시간 차트 업데이트"""
        detection_counts = data['detection_counts']
        current_detection = data['current_detection']
        elapsed_time = data['elapsed_time']
        frame_count = data['frame_count']
        
        # 1. 막대 그래프 - 탐지 갯수 표시
        self.ax1.clear()
        labels = list(detection_counts.keys())
        values = [detection_counts[k] for k in labels]
        colors = ['orange', 'skyblue', 'lightgray']
        
        bars = self.ax1.bar(labels, values, color=colors)
        chart_count = lang_kr.get('chart_count', '갯수')
        self.ax1.set_title(chart_count, fontsize=20, weight='bold')
        self.ax1.set_ylabel(chart_count, fontsize=16)
        self.ax1.tick_params(axis='x', labelsize=16)
        
        # 2. 파이 차트 - 탐지 비율 표시
        self.ax2.clear()
        total = sum(detection_counts.values())
        if total > 0:
            labels = list(detection_counts.keys())
            values = [detection_counts[k] for k in labels]
            colors = ['orange', 'skyblue', 'lightgray']
            
            self.ax2.pie(values, 
                         labels=labels, 
                         colors=colors, 
                         autopct='%1.1f%%', 
                         startangle=90, 
                         textprops={'fontsize': 12})
            chart_distribution = lang_kr.get('chart_distribution', '분포')
            self.ax2.set_title(f'{chart_distribution} ({total})', fontsize=20, weight='bold')

        # 3. 현재 상태 텍스트 (예약 공간)
        self.ax3.clear()
        self.ax3.axis('off')
                
        # 막대 그래프에 값 표시
        for bar, value in zip(bars, values):
            if value > 0:
                self.ax1.text(bar.get_x() + bar.get_width()/2, 
                             bar.get_height() + 0.5, 
                             str(value), ha='center', va='bottom', fontsize=16)

        # 4. FPS 및 성능 표시
        self.ax4.clear()
        
        # 현재 탐지 결과 텍스트 길이 제한
        detection_text = current_detection[:20] + "..." if len(current_detection) > 20 else current_detection
        
        # lang_kr에서 안전하게 값 가져오기
        current_text = lang_kr.get('current', '현재')
        time_text = lang_kr.get('time', '시간')
        processed_text = lang_kr.get('processed', '처리됨')
        detection_fps_text = lang_kr.get('detection_fps', '탐지 FPS')
        total_text = lang_kr.get('total', '전체')
        rate_text = lang_kr.get('rate', '비율')
        
        self.ax4.text(0.5, 0.9, f'{current_text}: {detection_text}', ha='center', va='center', fontsize=20, weight='bold')
        self.ax4.text(0.5, 0.8, f'{time_text}: {elapsed_time:.1f}초', ha='center', va='center', fontsize=16)
        self.ax4.text(0.5, 0.7, f'{processed_text}: {frame_count}', ha='center', va='center', fontsize=16)

        # 성능 지표 계산 및 표시
        if elapsed_time > 0:
            detection_fps = frame_count / elapsed_time
            self.ax4.text(0.5, 0.6, f'{detection_fps_text}: {detection_fps:.1f}', ha='center', va='center', fontsize=16, weight='bold')
            self.ax4.text(0.5, 0.5, f'{total_text}: {total}', ha='center', va='center', fontsize=16)
            if total > 0:
                detection_rate = (total - detection_counts['none']) / total * 100
                self.ax4.text(0.5, 0.3, f'{rate_text}: {detection_rate:.1f}%', ha='center', va='center', fontsize=16)

        chart_performance = lang_kr.get('chart_performance', '상태/성능')
        self.ax4.set_title(chart_performance, fontsize=20, weight='bold')
        self.ax4.axis('off')
        
        self.draw()

class YOLODetectionGUI(QMainWindow):
    """메인 GUI 클래스 - YOLO 실시간 탐지 인터페이스"""
    
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.yolo_thread = None
        self.detection_data = []
        self.start_time = None
        self.duration = 20  # 20초 자동 실행
        self.initUI()
        
    def initUI(self):
        """UI 초기화 - 전체 인터페이스 구성"""
        self.setWindowTitle(lang_kr['window_title'])
        self.setGeometry(100, 100, 1600, 1000)
        
        # 중앙 위젯 설정: 전체 GUI의 중심이 되는 QWidget 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 설정: 좌우로 패널을 배치하는 QHBoxLayout 사용
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽 패널 (컨트롤): 버튼, 상태, 영상 등 컨트롤 요소를 세로로 배치
        left_panel = QVBoxLayout()
        
        # 제목 라벨: 프로그램 이름 표시
        title_label = QLabel(lang_kr['title'])
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(title_label)
        
        # 탐지 시작 버튼: YOLO 탐지 시작
        self.start_button = QPushButton(lang_kr['start_btn'])
        self.start_button.clicked.connect(self.start_detection)
        left_panel.addWidget(self.start_button)
        
        # 탐지 중지 버튼: YOLO 탐지 중지
        self.stop_button = QPushButton(lang_kr['stop_btn'])
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        left_panel.addWidget(self.stop_button)
        
        # 진행 상황 표시: 탐지 진행률을 보여주는 ProgressBar
        self.progress_bar = QProgressBar()
        left_panel.addWidget(self.progress_bar)
        
        # 현재 상태 표시: 탐지 상태(준비/실행/완료) 표시
        self.status_label = QLabel(lang_kr['status_ready'])
        self.status_label.setFont(QFont("Arial", 12))
        left_panel.addWidget(self.status_label)
        
        # FPS 표시: 카메라 프레임 속도 표시
        self.fps_label = QLabel(lang_kr['camera_fps'])
        self.fps_label.setFont(QFont("Arial", 10))
        left_panel.addWidget(self.fps_label)
        
        # YOLO 처리 주기 표시: YOLO가 몇 초마다 처리되는지 안내
        self.yolo_timing_label = QLabel(lang_kr['yolo_timing'])
        self.yolo_timing_label.setFont(QFont("Arial", 10))
        self.yolo_timing_label.setStyleSheet("color: green")
        left_panel.addWidget(self.yolo_timing_label)
        
        # 현재 탐지 결과 표시: 실시간 탐지 결과 텍스트
        self.detection_label = QLabel(lang_kr['current_detection'])
        self.detection_label.setFont(QFont("Arial", 11))
        self.detection_label.setWordWrap(True)
        left_panel.addWidget(self.detection_label)
        
        # 라이브 카메라 영상 표시 (원본): 실시간 원본 영상 표시
        self.camera_label = QLabel(lang_kr['camera_label'])
        self.camera_label.setFixedSize(320, 240)
        self.camera_label.setStyleSheet("border: 2px solid blue")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setScaledContents(True)
        left_panel.addWidget(self.camera_label)
        
        # YOLO 결과 영상 표시: YOLO가 처리한 영상 표시
        self.yolo_label = QLabel(lang_kr['yolo_label'])
        self.yolo_label.setFixedSize(320, 240)
        self.yolo_label.setStyleSheet("border: 2px solid red")
        self.yolo_label.setAlignment(Qt.AlignCenter)
        self.yolo_label.setScaledContents(True)
        left_panel.addWidget(self.yolo_label)
        
        # 로그 텍스트: 탐지 로그 및 메시지 표시
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setFont(QFont("Arial", 8))
        self.log_text.setReadOnly(True)
        left_panel.addWidget(self.log_text)
        
        # 최종 결과 표시: 탐지 종료 후 결과 요약 표시
        self.result_label = QLabel()
        self.result_label.setFont(QFont("Arial", 8))
        self.result_label.setWordWrap(True)
        left_panel.addWidget(self.result_label)
        
        # 왼쪽 패널을 QWidget으로 래핑하여 너비 제한
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(350)
        
        # 오른쪽 패널 (통계 차트): 탐지 통계 차트 표시용 캔버스
        self.stats_canvas = StatisticsCanvas()
        
        # 메인 레이아웃에 좌측 컨트롤 패널과 우측 차트 패널 추가
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.stats_canvas)
        
        # 타이머 설정: UI 진행 상황 업데이트용 (1초마다)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        
        # FPS 계산용 변수 초기화
        self.camera_frame_count = 0
        self.last_fps_time = time.time()
        
    def cv2_to_qpixmap(self, cv_img):
        """OpenCV 이미지를 QPixmap으로 변환 (Qt 표시용)"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)
        
    def start_detection(self):
        """탐지 시작 - 카메라 및 YOLO 스레드 시작"""
        model_path = r"D:\GoogleDrive\modeling_yolo\yolo_20250906_205051\weights\best.pt"
        #model_path = r"D:\GoogleDrive\modeling_yolo\yolo_20250907_112223\weights\best.pt"
        
        # 카메라 스레드 시작
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_camera_feed)
        self.camera_thread.start()
        
        # YOLO 스레드 시작
        self.yolo_thread = YOLODetectionThread(model_path)
        self.yolo_thread.detection_signal.connect(self.update_detection)
        self.yolo_thread.start()
        
        # UI 상태 업데이트
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(lang_kr['status_running'])
        self.progress_bar.setRange(0, self.duration)
        self.timer.start(1000)  # 1초마다 업데이트
        
        # 로그 및 데이터 초기화
        self.log_text.append(lang_kr['log_start'])
        self.log_text.append(lang_kr['log_info'])
        self.detection_data = []
        self.start_time = time.time()
        self.camera_frame_count = 0
        self.last_fps_time = time.time()
        
    def stop_detection(self):
        """탐지 중지 - 모든 스레드 정리"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait(3000)  # 3초 대기
            
        if self.yolo_thread:
            self.yolo_thread.stop()
            self.yolo_thread.wait(3000)  # 3초 대기
        
        self.detection_finished()
        
    def update_camera_feed(self, frame):
        """카메라 피드 업데이트 (고속, 부드러운 영상)"""
        # 원본 카메라 영상 표시
        pixmap = self.cv2_to_qpixmap(frame)
        self.camera_label.setPixmap(pixmap)
        
        # FPS 계산 및 표시
        self.camera_frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # 1초마다 FPS 업데이트
            fps = self.camera_frame_count / (current_time - self.last_fps_time)
            self.fps_label.setText(f"{lang_kr['camera_fps'][:-1]}: {fps:.1f}")
            self.camera_frame_count = 0
            self.last_fps_time = current_time
        
        # YOLO 스레드로 프레임 전송 (0.2초 주기로 자동 제한됨)
        if self.yolo_thread:
            self.yolo_thread.add_frame(frame)
        
        # 시간 체크 (자동 종료)
        if self.start_time and (current_time - self.start_time) > self.duration:
            self.stop_detection()
        
    def update_detection(self, data):
        """YOLO 탐지 결과 업데이트"""
        self.detection_data.append(data)
        
        # 현재 탐지 결과 표시
        self.detection_label.setText(f"{lang_kr['current_detection'][:-1]}: {data['current_detection']}")
        
        # YOLO 결과 영상 표시
        if 'annotated_frame' in data and data['annotated_frame'] is not None:
            pixmap = self.cv2_to_qpixmap(data['annotated_frame'])
            self.yolo_label.setPixmap(pixmap)
        
        # 로그 업데이트 (탐지가 있을 때만)
        if data['current_detection'] != "No Detection":
            self.log_text.append(f"{data['frame_count']}{lang_kr['frame_suffix']}: {data['current_detection']}")
            # 스크롤을 맨 아래로
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum())
        
        # 차트 업데이트
        self.stats_canvas.update_charts(data)

    def update_progress(self):
        """진행 상황 업데이트"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.progress_bar.setValue(int(elapsed))
            self.status_label.setText(f"{lang_kr['status_running']} ({elapsed:.1f}초)")
        
    def detection_finished(self):
        """탐지 완료 - 결과 정리 및 표시"""
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(lang_kr['status_done'])
        self.progress_bar.setValue(self.duration)
        
        # 영상 초기화
        self.camera_label.setText(lang_kr['camera_label'])
        #self.camera_label.setPixmap(QPixmap())
        self.yolo_label.setText(lang_kr['yolo_label'])
        #self.yolo_label.setPixmap(QPixmap())
        self.fps_label.setText(lang_kr['camera_fps'])
        
        # 최종 결과 계산 및 표시
        if self.detection_data:
            final_data = self.detection_data[-1]
            detection_counts = final_data['detection_counts']
            total = sum(detection_counts.values())
            elapsed = final_data['elapsed_time']
            detection_fps = final_data['frame_count'] / elapsed if elapsed > 0 else 0
            
            result_text = f"""{lang_kr['final_result']}:
{lang_kr['run_time']}: {elapsed:.1f}초
{lang_kr['detect_fps']}: {detection_fps:.1f}
{lang_kr['frame_processed']}: {final_data['frame_count']}

{lang_kr['detect_summary']}:"""
            
            for class_name, count in detection_counts.items():
                if total > 0:
                    percentage = count/total*100
                    result_text += f"\n{class_name}: {count} ({percentage:.1f}%)"
            
            # 차트에 최종 결과 표시
            self.stats_canvas.ax3.clear()
            self.stats_canvas.ax3.text(0.05, 0.95, lang_kr['final_result'], ha='left', va='top', fontsize=20, weight='bold')
            self.stats_canvas.ax3.text(0.05, 0.85, result_text, ha='left', va='top', fontsize=18)
            self.stats_canvas.ax3.axis('off')
            self.stats_canvas.draw()            

            self.log_text.append(lang_kr['log_done'])

def main():
    """메인 함수 - 애플리케이션 시작"""
    app = QApplication(sys.argv)
    window = YOLODetectionGUI()
    window.show()
    sys.exit(app.exec_())

# 한글 텍스트 할당
lang_kr = {
    'window_title': 'YOLO 실시간 반려동물 탐지 - 최적화',
    'title': 'YOLO 탐지',
    'start_btn': '탐지 시작 (20초)',
    'stop_btn': '탐지 중지',
    'status_ready': '상태: 준비됨',
    'status_running': '상태: 실행 중...',
    'status_done': '상태: 완료됨',
    'camera_fps': '카메라 FPS: -',
    'yolo_timing': 'YOLO: 0.2초마다',
    'current_detection': '현재 탐지: -',
    'camera_label': '실시간 카메라 영상 (30 FPS)',
    'yolo_label': 'YOLO 결과 영상 (5 FPS)',
    'log_start': '탐지를 시작합니다...',
    'log_info': '카메라: 30 FPS, YOLO: 0.2초마다 (5 FPS)',
    'log_done': '탐지가 완료되었습니다.',
    'frame_suffix': '번째 프레임',
    'final_result': '최종 결과',
    'run_time': '실행 시간',
    'detect_fps': '탐지 FPS',
    'frame_processed': '처리된 프레임',
    'detect_summary': '탐지 요약',
    'chart_title': 'YOLO 탐지 통계',
    'chart_count': '갯수',
    'chart_distribution': '분포',
    'chart_performance': '상태/성능',
    'current': '현재',
    'time': '시간',
    'processed': '처리됨',
    'detection_fps': '탐지 FPS',
    'total': '전체',
    'rate': '비율'
}

if __name__ == '__main__':
    main()