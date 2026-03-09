# services/mmseg_service.py
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

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

        # ------- 게이트/스코어 파라미터(init) -------
        self.MIN_PLATE_AREA_RATIO = 0.015  
        self.MIN_PLATE_PIXELS = 1500       

        # - severe: 거의 붙어서 잘린 건 RETAKE
        # - warn: 살짝 타이트한 건 OK + 경고만
        self.CROP_SEVERE_MARGIN_RATIO = 0.001  # 0.1% 이하면 retake
        self.CROP_WARN_MARGIN_RATIO = 0.004    # 0.4% 이하면 경고

        # 양념 번짐 눌러주는 weighted ratio
        self.USE_WEIGHTED = True
        self.DIST_TAU = 12                 # leftover 두께 기준(px) (해상도 따라 조절)
        self.W_EPS = 0.08                  # 번짐도 아주 조금은 잔반으로 치고 싶으면 0.05~0.15


    def _shot_quality_gate(self, pred: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        H, W = pred.shape
        img_area = H * W

        plate = (pred == self.PLATE)
        lo = (pred == self.LEFTOVER)

        plate_pixels = int(plate.sum())
        plate_area_ratio = plate_pixels / float(img_area + 1e-6)

        if plate_pixels <= 0:
            return False, {"reason": "no_plate", "plate_pixels": plate_pixels}

        # 픽셀 기반 최소치(해상도/거리 영향 덜 받게 완화)
        if plate_pixels < self.MIN_PLATE_PIXELS:
            return False, {
                "reason": "plate_too_small_px",
                "plate_pixels": plate_pixels,
                "plate_area_ratio": plate_area_ratio,
                "min_plate_pixels": self.MIN_PLATE_PIXELS,
            }

        # 면적 비율 최소치(완화)
        if plate_area_ratio < self.MIN_PLATE_AREA_RATIO:
            return False, {
                "reason": "plate_too_small_area",
                "plate_pixels": plate_pixels,
                "plate_area_ratio": plate_area_ratio,
                "min_plate_area_ratio": self.MIN_PLATE_AREA_RATIO,
            }
        # ---- 크롭 판정: bbox margin ----
        area_mask = (plate | lo)
        ys, xs = np.where(area_mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        top_margin = y0
        left_margin = x0
        bottom_margin = (H - 1) - y1
        right_margin = (W - 1) - x1
        min_margin = int(min(top_margin, left_margin, bottom_margin, right_margin))

        # 해상도 비율 기반 threshold
        severe_thr = int(self.CROP_SEVERE_MARGIN_RATIO * min(H, W))
        warn_thr = int(self.CROP_WARN_MARGIN_RATIO * min(H, W))

        if min_margin <= severe_thr:
            # RETAKE
            return False, {
                "reason": "cropped_severe",
                "bbox": [x0, y0, x1, y1],
                "min_margin": min_margin,
                "severe_thr": severe_thr,
                "warn_thr": warn_thr,
                "plate_area_ratio": plate_area_ratio,
                "plate_pixels": plate_pixels,
            }

        total = int((plate | lo).sum())
        total_area_ratio = total / float(img_area + 1e-6)

        if min_margin <= warn_thr:
            # OK는 주되 diag로 경고만 남김
            return True, {
                "reason": "cropped_warn",
                "bbox": [x0, y0, x1, y1],
                "min_margin": min_margin,
                "warn_thr": warn_thr,
                "plate_area_ratio": plate_area_ratio,
                "plate_pixels": plate_pixels,
                "total_area_ratio": total_area_ratio,
            }

        return True, {
            "reason": "ok",
            "bbox": [x0, y0, x1, y1],
            "min_margin": min_margin,
            "plate_area_ratio": plate_area_ratio,
            "plate_pixels": plate_pixels,
            "total_area_ratio": total_area_ratio,
        }

    def _weighted_leftover_ratio(self, img_rgb: np.ndarray, pred: np.ndarray) -> float:
        plate = (pred == self.PLATE)
        lo = (pred == self.LEFTOVER)

        denom = int((plate | lo).sum())
        if denom == 0:
            return 0.0

        try:
            import cv2  # type: ignore
            lo_u8 = lo.astype(np.uint8)
            dist = cv2.distanceTransform(lo_u8, distanceType=cv2.DIST_L2, maskSize=3)
            w_thin = np.clip(dist / float(self.DIST_TAU), 0.0, 1.0)
        except Exception:
            w_thin = np.ones_like(pred, dtype=np.float32)

        w = np.maximum(self.W_EPS, w_thin).astype(np.float32)
        numer = float((w * lo).sum())
        return numer / float(denom)

    def calculate_leftover_ratio(self, image: Image.Image) -> float:
        img_np = np.array(image.convert("RGB"))
        result = inference_model(self.model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)

        ok, _diag = self._shot_quality_gate(pred)
        if not ok:
            return 0.0

        # calculate_leftover_ratio
        if self.USE_WEIGHTED:
            return 1.0 - float(self._weighted_leftover_ratio(img_np, pred))

        leftover_pixels = int((pred == self.LEFTOVER).sum())
        plate_pixels = int((pred == self.PLATE).sum())
        total = leftover_pixels + plate_pixels
        return 0.0 if total == 0 else (1.0 - (leftover_pixels / float(total)))

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        img_np = np.array(image.convert("RGB"))
        result = inference_model(self.model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)

        ok, diag = self._shot_quality_gate(pred)
        reason = diag.get("reason", "unknown")

        # ratio는 가능한 한 계산
        if self.USE_WEIGHTED:
            ratio = 1.0 - float(self._weighted_leftover_ratio(img_np, pred))
        else:
            leftover_pixels = int((pred == self.LEFTOVER).sum())
            plate_pixels = int((pred == self.PLATE).sum())
            total = leftover_pixels + plate_pixels
            ratio = 0.0 if total == 0 else (1.0 - (leftover_pixels / float(total)))

        # RETAKE는 진짜로 plate가 전혀 없을 때
        if not ok and reason == "no_plate":
            return {
                "status": "RETAKE",
                "leftover_ratio": None,
                "message": "그릇(식판)을 화면 중앙에 두고 다시 촬영해주세요.",
                "diag": diag,
            }

        # 나머지 실패들은: OK + 경고 
        warn_msgs = {
            "plate_too_small_px": "그릇(식판) 인식이 작게 잡혔어요. 결과 정확도가 낮을 수 있어요.",
            "plate_too_small_area": "그릇(식판) 인식이 작게 잡혔어요. 결과 정확도가 낮을 수 있어요.",
            "cropped_severe": "식판이 화면 가장자리에 가까워 보여요. 다음엔 여백을 조금만 더 주세요.",
            "cropped_warn": "식판이 화면에 조금 타이트해요. 다음엔 여백을 조금만 더 두면 더 정확해요.",
        }

        msg = warn_msgs.get(reason)

        resp = {"status": "OK", "leftover_ratio": float(ratio), "diag": diag}
        if msg:
            resp["message"] = msg
            resp["low_confidence"] = True  
        return resp