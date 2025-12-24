import torch
import io
import base64
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from ts.context import Context


class DiffusersHandler:
    def initialize(self, context):
        """
        모델 및 파이프라인 초기화
        Text-to-Image와 Image-to-Image 파이프라인을 모두 로드하되,
        메모리 절약을 위해 구성 요소를 공유합니다.
        """
        # 1. Text-to-Image 파이프라인 로드 (기존과 동일)
        # L4 GPU 최적화: float16 사용
        self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        # 가속을 위한 메모리 최적화 (xformers)
        self.txt2img_pipe.enable_xformers_memory_efficient_attention()

        # 2. Image-to-Image 파이프라인 로드
        # 중요: 메모리 절약을 위해 첫 번째 파이프라인의 구성 요소를 재사용합니다.
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        ).to("cuda")

        # img2img 파이프라인에도 메모리 최적화 적용
        self.img2img_pipe.enable_xformers_memory_efficient_attention()

        print("SDXL Handler initialized with txt2img and img2img support.")

    def handle(self, data, context):
        """
        요청 처리 메인 메서드
        입력 데이터에 따라 txt2img, img2img, img2img(유사) 중 선택 실행합니다.
        """
        if not data:
            return None

        row = data[0]
        payload = row.get("data") or row.get("body")

        if isinstance(payload, dict):
            params = payload
        else:
            import json

            params = json.loads(payload)

        prompt = params.get("prompt")
        input_image_b64 = params.get("image")

        # 분기 처리
        if input_image_b64 and prompt:
            print("Processing Image-to-Image request (with text guidance)...")
            # === Image + Text: Image-to-Image 변경 ===
            try:
                image_data = base64.b64decode(input_image_b64)
                init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                strength = float(params.get("strength", 0.75))

                output = self.img2img_pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                )
            except Exception as e:
                print(f"Error during img2img processing: {e}")
                return None

        elif input_image_b64 and not prompt:
            print("Processing Image-to-Image request (similar image)...")
            # === Image만: 유사한 이미지 생성 ===
            try:
                image_data = base64.b64decode(input_image_b64)
                init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

                # 프롬프트 없이 약간의 변형된 이미지 생성 (strength 낮게)
                strength = float(params.get("strength", 0.3))  # 기본값: 원본과 유사

                output = self.img2img_pipe(
                    prompt="",  # 빈 프롬프트
                    image=init_image,
                    strength=strength,
                )
            except Exception as e:
                print(f"Error during img2img (similar) processing: {e}")
                return None

        elif prompt and not input_image_b64:
            print("Processing Text-to-Image request...")
            # === Text만: Text-to-Image ===
            output = self.txt2img_pipe(prompt=prompt)

        else:
            print("Error: Either prompt or image is required.")
            return None

        # 결과 이미지 후처리
        image = output.images[0]
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="PNG")
        return [byte_arr.getvalue()]
