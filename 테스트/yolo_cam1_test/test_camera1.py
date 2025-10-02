import cv2
import matplotlib.pyplot as plt
import time

def test_camera1():
    """카메라 1번 테스트"""
    print("=== 카메라 1번 DirectShow 테스트 ===")
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ 카메라 1번을 열 수 없습니다.")
        return
    
    # 카메라 속성 확인
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 카메라 1번 초기 설정: {width}x{height}, FPS: {fps}")
    
    # 몇 프레임 건너뛰기
    for i in range(5):
        ret, frame = cap.read()
        print(f"  프레임 {i+1}: ret={ret}, frame={'None' if frame is None else frame.shape}")
        if ret and frame is not None:
            print(f"  프레임 통계: min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
        time.sleep(0.2)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        print(f"✅ 카메라 1번 성공! 크기: {frame.shape}")
        print(f"✅ 프레임 통계: min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
        
        # BGR -> RGB 변환 후 matplotlib로 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.title('Camera 1 Preview (DirectShow)')
        plt.tight_layout()
        plt.show()
    else:
        print("❌ 카메라 1번 프레임 읽기 실패")

if __name__ == "__main__":
    test_camera1()
