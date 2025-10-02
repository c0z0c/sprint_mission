import cv2
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

def camera1_preview_live():
    """카메라 1번으로 실시간 프리뷰"""
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ 카메라 1번을 열 수 없습니다.")
        return
    
    print("✅ 카메라 1번 실시간 프리뷰 시작!")
    print("Jupyter에서 커널 인터럽트(중지 버튼)로 종료하세요.")
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break
            
            # 5프레임마다 화면 업데이트 (부하 감소)
            if frame_count % 5 == 0:
                # BGR -> RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 화면 지우고 새로운 프레임 표시
                clear_output(wait=True)
                
                plt.figure(figsize=(12, 8))
                plt.imshow(frame_rgb)
                plt.axis('off')
                plt.title(f'Camera 1 Live Preview (Frame: {frame_count})', fontsize=14)
                plt.tight_layout()
                plt.show()
                
                # 짧은 대기
                time.sleep(0.1)
            
            frame_count += 1
            
            # 500프레임마다 상태 출력
            if frame_count % 500 == 0:
                print(f"프레임 {frame_count} 처리됨")
                
    except KeyboardInterrupt:
        print("사용자에 의해 중단되었습니다.")
    finally:
        cap.release()
        plt.close('all')
        print("카메라 해제 완료")

if __name__ == "__main__":
    camera1_preview_live()
