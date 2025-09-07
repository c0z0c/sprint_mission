# pip install PyQt5 matplotlib ultralytics opencv-python

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

class CameraThread(QThread):
    """카메라 영상 캡처를 위한 스레드 (고속)"""
    frame_signal = pyqtSignal(np.ndarray)  # 원본 프레임 전송
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = None
        
    def run(self):
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 30 FPS로 설정
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기를 1로 설정하여 지연 최소화
        
        print("카메라 초기화 완료")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
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
        """새 프레임 추가 (0.2초마다만 처리)"""
        current_time = time.time()
        
        # 0.2초(200ms)가 지났는지 확인
        if current_time - self.last_process_time >= 0.2:
            try:
                # 큐에 기존 프레임이 있으면 제거하고 새 프레임 추가
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
        model = YOLO(self.model_path)
        print("YOLO model loaded successfully")
        
        try:
            while self.running:
                try:
                    # 큐에서 프레임 가져오기 (0.3초 대기)
                    frame = self.frame_queue.get(timeout=0.3)
                    
                    # YOLO 탐지 수행
                    results = model(frame, verbose=False, conf=0.6)
                    current_time = time.time() - self.start_time
                    
                    # YOLO 결과를 프레임에 그리기
                    annotated_frame = results[0].plot()
                    
                    # 탐지 결과 처리
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
        self.fig = Figure(figsize=(8, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        
        # 서브플롯 생성
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        
        self.fig.suptitle('YOLO Detection Statistics', fontsize=20, weight='bold')
        self.fig.tight_layout()
        
    def update_charts(self, data):
        """차트 업데이트"""
        detection_counts = data['detection_counts']
        current_detection = data['current_detection']
        elapsed_time = data['elapsed_time']
        frame_count = data['frame_count']
        
        # 1. 막대 그래프
        self.ax1.clear()
        labels = list(detection_counts.keys())
        values = [detection_counts[k] for k in labels]
        colors = ['orange', 'skyblue', 'lightgray']
        
        bars = self.ax1.bar(labels, values, color=colors)
        self.ax1.set_title('Count', fontsize=20, weight='bold')
        self.ax1.set_ylabel('Count', fontsize=16)
        self.ax1.tick_params(axis='x', labelsize=16)
        
        # 2. 파이 차트
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
                         textprops={'fontsize': 16})
            self.ax2.set_title(f'Distribution ({total})', fontsize=20, weight='bold')

        # 3. 현재 상태 텍스트
        self.ax3.clear()
        # self.ax3.text(0.5, 0.7, f'Current: {detection_text}', ha='center', va='center', fontsize=9, weight='bold')
        # self.ax3.text(0.5, 0.5, f'Time: {elapsed_time:.1f}s', ha='center', va='center', fontsize=9)
        # self.ax3.text(0.5, 0.3, f'Processed: {frame_count}', ha='center', va='center', fontsize=9)
        # self.ax3.set_title('Status', fontsize=10)
        self.ax3.axis('off')
                
        # 값 표시
        for bar, value in zip(bars, values):
            if value > 0:
                self.ax3.text(bar.get_x() + bar.get_width()/2, 
                             bar.get_height() + 0.5, 
                             str(value), ha='center', va='bottom', fontsize=8)
        
        # 4. FPS 및 성능 표시
        self.ax4.clear()
        
        detection_text = current_detection[:20] + "..." if len(current_detection) > 20 else current_detection
        self.ax4.text(0.5, 0.9, f'Current: {detection_text}', ha='center', va='center', fontsize=16, weight='bold')
        self.ax4.text(0.5, 0.8, f'Time: {elapsed_time:.1f}s', ha='center', va='center', fontsize=16)
        self.ax4.text(0.5, 0.7, f'Processed: {frame_count}', ha='center', va='center', fontsize=16)

        if elapsed_time > 0:
            detection_fps = frame_count / elapsed_time
            self.ax4.text(0.5, 0.6, f'Detection FPS: {detection_fps:.1f}', ha='center', va='center', fontsize=16, weight='bold')
            self.ax4.text(0.5, 0.5, f'Total: {total}', ha='center', va='center', fontsize=16)
            if total > 0:
                detection_rate = (total - detection_counts['none']) / total * 100
                self.ax4.text(0.5, 0.3, f'Rate: {detection_rate:.1f}%', ha='center', va='center', fontsize=16)
        self.ax4.set_title('Status/Performance', fontsize=20, weight='bold')
        self.ax4.axis('off')
        
        self.draw()

class YOLODetectionGUI(QMainWindow):
    """메인 GUI 클래스"""
    
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.yolo_thread = None
        self.detection_data = []
        self.start_time = None
        self.duration = 20  # 20초
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('YOLO Real-time Pet Detection - Optimized')
        self.setGeometry(100, 100, 1600, 1000)
        
        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 설정
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽 패널 (컨트롤)
        left_panel = QVBoxLayout()
        
        # 제목
        title_label = QLabel("YOLO Detection")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(title_label)
        
        # 시작/중지 버튼
        self.start_button = QPushButton("Start Detection (20sec)")
        self.start_button.clicked.connect(self.start_detection)
        left_panel.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        left_panel.addWidget(self.stop_button)
        
        # 진행 상황 표시
        self.progress_bar = QProgressBar()
        left_panel.addWidget(self.progress_bar)
        
        # 현재 상태 표시
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 12))
        left_panel.addWidget(self.status_label)
        
        # FPS 표시
        self.fps_label = QLabel("Camera FPS: -")
        self.fps_label.setFont(QFont("Arial", 10))
        left_panel.addWidget(self.fps_label)
        
        # YOLO 처리 주기 표시
        self.yolo_timing_label = QLabel("YOLO: Every 0.2s")
        self.yolo_timing_label.setFont(QFont("Arial", 10))
        self.yolo_timing_label.setStyleSheet("color: green")
        left_panel.addWidget(self.yolo_timing_label)
        
        # 현재 탐지 결과 표시
        self.detection_label = QLabel("Current Detection: -")
        self.detection_label.setFont(QFont("Arial", 11))
        self.detection_label.setWordWrap(True)
        left_panel.addWidget(self.detection_label)
        
        # 라이브 카메라 영상 표시 (원본)
        self.camera_label = QLabel("Live Camera Feed (30 FPS)")
        self.camera_label.setFixedSize(320, 240)
        self.camera_label.setStyleSheet("border: 2px solid blue")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setScaledContents(True)
        left_panel.addWidget(self.camera_label)
        
        # YOLO 결과 영상 표시
        self.yolo_label = QLabel("YOLO Result (5 FPS)")
        self.yolo_label.setFixedSize(320, 240)
        self.yolo_label.setStyleSheet("border: 2px solid red")
        self.yolo_label.setAlignment(Qt.AlignCenter)
        self.yolo_label.setScaledContents(True)
        left_panel.addWidget(self.yolo_label)
        
        # 로그 텍스트
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setFont(QFont("Arial", 8))
        self.log_text.setReadOnly(True)
        left_panel.addWidget(self.log_text)
        
        # 최종 결과 표시
#         self.result_label = QLabel(
# """Final Results:
# Runtime: 0s
# Detection FPS: 0
# Processed Frames: 0

# Detection Summary:


# """
# )
        self.result_label = QLabel()
        self.result_label.setFont(QFont("Arial", 8))
        self.result_label.setWordWrap(True)
        left_panel.addWidget(self.result_label)
        

        # 왼쪽 패널을 위젯으로 만들기
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(350)
        
        # 오른쪽 패널 (통계 차트)
        self.stats_canvas = StatisticsCanvas()
        
        # 메인 레이아웃에 추가
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.stats_canvas)
        
        # 타이머 설정 (UI 업데이트용)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        
        # FPS 계산용 변수
        self.camera_frame_count = 0
        self.last_fps_time = time.time()
        
    def cv2_to_qpixmap(self, cv_img):
        """OpenCV 이미지를 QPixmap으로 변환"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)
        
    def start_detection(self):
        """탐지 시작"""
        model_path = r"D:\GoogleDrive\modeling_yolo\yolo_20250906_205051\weights\best.pt"
        
        # 카메라 스레드 시작
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_camera_feed)
        self.camera_thread.start()
        
        # YOLO 스레드 시작
        self.yolo_thread = YOLODetectionThread(model_path)
        self.yolo_thread.detection_signal.connect(self.update_detection)
        self.yolo_thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Running...")
        self.progress_bar.setRange(0, self.duration)
        self.timer.start(1000)  # 1초마다 업데이트
        
        self.log_text.append("Detection started...")
        self.log_text.append("Camera: 30 FPS, YOLO: Every 0.2s (5 FPS)")
        self.detection_data = []
        self.start_time = time.time()
        self.camera_frame_count = 0
        self.last_fps_time = time.time()
        
    def stop_detection(self):
        """탐지 중지"""
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
        
        # FPS 계산
        self.camera_frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # 1초마다 FPS 업데이트
            fps = self.camera_frame_count / (current_time - self.last_fps_time)
            self.fps_label.setText(f"Camera FPS: {fps:.1f}")
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
        self.detection_label.setText(f"Current: {data['current_detection']}")
        
        # YOLO 결과 영상 표시
        if 'annotated_frame' in data and data['annotated_frame'] is not None:
            pixmap = self.cv2_to_qpixmap(data['annotated_frame'])
            self.yolo_label.setPixmap(pixmap)
        
        # 로그 업데이트 (탐지가 있을 때만)
        if data['current_detection'] != "No Detection":
            self.log_text.append(f"Frame {data['frame_count']}: {data['current_detection']}")
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
            self.status_label.setText(f"Status: Running... ({elapsed:.1f}s)")
        
    def detection_finished(self):
        """탐지 완료"""
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Finished")
        self.progress_bar.setValue(self.duration)
        
        # 영상 초기화
        self.camera_label.setText("Live Camera Feed (30 FPS)")
        self.camera_label.setPixmap(QPixmap())
        self.yolo_label.setText("YOLO Result (5 FPS)")
        self.yolo_label.setPixmap(QPixmap())
        self.fps_label.setText("Camera FPS: -")
        
        # 최종 결과 계산 및 표시
        if self.detection_data:
            final_data = self.detection_data[-1]
            detection_counts = final_data['detection_counts']
            total = sum(detection_counts.values())
            elapsed = final_data['elapsed_time']
            detection_fps = final_data['frame_count'] / elapsed if elapsed > 0 else 0
            
            result_text = f"""Final Results:
Runtime: {elapsed:.1f}s
Detection FPS: {detection_fps:.1f}
Processed Frames: {final_data['frame_count']}

Detection Summary:"""
            
            for class_name, count in detection_counts.items():
                if total > 0:
                    percentage = count/total*100
                    result_text += f"\n{class_name}: {count} ({percentage:.1f}%)"
            
            
            self.stats_canvas.ax3.clear()
            # self.stats_canvas.ax3.text(0.5, 0.7, f'Current: {detection_text}', ha='center', va='center', fontsize=9, weight='bold')
            # self.stats_canvas.ax3.text(0.5, 0.5, f'Time: {elapsed_time:.1f}s', ha='center', va='center', fontsize=9)
            # self.stats_canvas.ax3.text(0.5, 0.3, f'Processed: {frame_count}', ha='center', va='center', fontsize=9)
            # seself.stats_canvaslf.ax3.set_title('Status', fontsize=10)

            # self.stats_canvas.ax3.text(0.5, 0.5, result_text, ha='center', va='center', fontsize=9)
            # self.stats_canvas.ax3.axis('off')
            # self.stats_canvas.draw()
            
            self.stats_canvas.ax3.text(0.05, 0.95, "Final Results", ha='left', va='top', fontsize=20, weight='bold')
            self.stats_canvas.ax3.text(0.05, 0.85, result_text, ha='left', va='top', fontsize=18)
            self.stats_canvas.ax3.axis('off')
            self.stats_canvas.draw()            

            #self.result_label.setText(result_text)
            self.log_text.append("Detection Completed")

def main():
    app = QApplication(sys.argv)
    window = YOLODetectionGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()