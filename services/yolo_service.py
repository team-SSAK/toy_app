import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

class YOLOService:
    """YOLO 모델을 사용한 잔반 비율 계산 서비스"""
    
    def __init__(self, model_path: str):
        """
        YOLO 모델 초기화
        
        Args:
            model_path: YOLO 모델 파일 경로
        """
        self.model = YOLO(model_path)
    
    def calculate_leftover_ratio(self, image: Image.Image) -> float:
        """
        잔반 비율 계산
        
        필터링 규칙:
        - 'dishes'는 항상 포함
        - 'leftovers'는 신뢰도 0.4 이상일 때만 포함
        
        Args:
            image: PIL Image 객체
        
        Returns:
            잔반 비율 (0.0 ~ 1.0)
        """
        results = self.model(image, verbose=False)
        result = results[0]

        if result.masks is None or result.boxes is None:
            return 0.0

        h, w = result.orig_shape
        leftover_mask = np.zeros((h, w), dtype=bool)
        dish_mask = np.zeros((h, w), dtype=bool)

        for i in range(len(result.boxes)):
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
            confidence = float(result.boxes.conf[i])

            # 필터링 로직
            is_kept = False
            if class_name == "dishes":
                is_kept = True
            elif class_name == "leftovers" and confidence >= 0.4:
                is_kept = True
            
            if not is_kept:
                continue

            # 마스크 처리
            instance_mask_tensor = result.masks.data[i]
            instance_mask_np = instance_mask_tensor.cpu().numpy().astype(np.uint8)
            instance_mask_resized = cv2.resize(instance_mask_np, (w, h)).astype(bool)

            if class_name == "leftovers":
                leftover_mask |= instance_mask_resized
            elif class_name == "dishes":
                dish_mask |= instance_mask_resized

        # 겹치는 영역 처리: 그릇 영역에서 잔반 영역 제외
        final_dish_mask = dish_mask & (~leftover_mask)

        leftover_pixels = np.sum(leftover_mask)
        dishes_pixels = np.sum(final_dish_mask)
        total_dish_area = leftover_pixels + dishes_pixels

        if total_dish_area == 0:
            return 0.0

        ratio = (leftover_pixels / total_dish_area)
        return ratio