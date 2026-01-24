# services/mmseg_service.py
import sys
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from mmseg.apis import init_model, inference_model


class MMSegService:
    """MMSegmentation(Mask2Former) 기반 잔반 비율 계산 서비스"""

    def __init__(self, repo_root: str, checkpoint_path: str, device: str | None = None):
        """
        Args:
            repo_root: models/segmentation_plate_leftover-main (configs/, mmseg/ 있는 폴더)
            checkpoint_path: models/best_mIoU_iter_12000.pth
            device: "cuda:0" or "cpu" (None이면 자동)
        """
        self.repo_root = Path(repo_root).resolve()
        self.checkpoint_path = str(Path(checkpoint_path).resolve())

        # mmseg import가 되도록 repo_root를 sys.path에 추가 (repo_root 안에 mmseg/ 폴더가 있음)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        # config 경로 (문서에 적힌 그 파일)
        self.config_path = str(
            self.repo_root / "configs" / "mask2former" / "mask2former_swin-s_food-dish.py"
        )

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 모델 로드 (서버 시작 시 1번만)
        self.model = init_model(self.config_path, self.checkpoint_path, device=self.device)

        # 클래스 인덱스(보통 0=bg, 1=plate, 2=leftover) — 만약 결과 이상하면 여기만 바꾸면 됨
        self.BG = 0
        self.PLATE = 1
        self.LEFTOVER = 2

    def calculate_leftover_ratio(self, image: Image.Image) -> float:
        """
        Returns:
            잔반 비율 (0.0 ~ 1.0)
            = leftover_pixels / (plate_pixels + leftover_pixels)
        """
        # mmseg는 파일 경로로 inference 하는 게 제일 안정적이라
        # 여기서는 PIL 이미지를 numpy로 바꿔서 바로 넣는 방식 사용(대부분 동작)
        img_np = np.array(image.convert("RGB"))

        result = inference_model(self.model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)  # (H, W)

        leftover_pixels = int((pred == self.LEFTOVER).sum())
        plate_pixels = int((pred == self.PLATE).sum())
        total = leftover_pixels + plate_pixels

        if total == 0:
            return 0.0

        return leftover_pixels / total
