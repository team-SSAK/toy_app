# services/mmseg_service.py
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

import torch
from mmseg.apis import init_model, inference_model


class MMSegService:
    """MMSegmentation(Mask2Former) 기반 비움 비율 계산 서비스"""

    def __init__(self, repo_root: str, checkpoint_path: str, device: str | None = None):
        """
        Args:
            repo_root: models/segmentation_plate_leftover-main (configs/, mmseg/ 있는 폴더)
            checkpoint_path: 사용할 checkpoint 경로
            device: "cuda:0" or "cpu" (None이면 자동)
        """
        self.repo_root = Path(repo_root).resolve()
        self.checkpoint_path = str(Path(checkpoint_path).resolve())

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        self.config_path = str(
            self.repo_root / "configs" / "mask2former" / "mask2former_swin-s_food-dish.py"
        )

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = init_model(self.config_path, self.checkpoint_path, device=self.device)

        # 클래스 인덱스 확정:
        # 0 = plate, 1 = leftover, 2 = background
        self.PLATE = 0
        self.LEFTOVER = 1
        self.BG = 2

        # ------- 게이트/스코어 파라미터 -------
        self.MIN_PLATE_AREA_RATIO = 0.015
        self.MIN_PLATE_PIXELS = 1500

        # severe: 거의 붙어서 잘린 건 RETAKE
        # warn: 살짝 타이트한 건 OK + 경고만
        self.CROP_SEVERE_MARGIN_RATIO = 0.001
        self.CROP_WARN_MARGIN_RATIO = 0.004

        # 잔반 가중치 계산
        self.USE_WEIGHTED = True
        self.DIST_TAU = 12
        self.W_EPS = 0.08

        # 작은 잔반 노이즈 제거
        self.MIN_LEFTOVER_AREA_RATIO = 0.005   # plate 면적의 0.5% 미만 blob 제거
        self.EMPTY_RATIO_SNAP = 0.95           # 95% 이상 비우면 100%로 스냅

    def _shot_quality_gate(self, pred: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        H, W = pred.shape
        img_area = H * W

        plate = (pred == self.PLATE)
        lo = (pred == self.LEFTOVER)

        plate_pixels = int(plate.sum())
        plate_area_ratio = plate_pixels / float(img_area + 1e-6)

        if plate_pixels <= 0:
            return False, {"reason": "no_plate", "plate_pixels": plate_pixels}

        if plate_pixels < self.MIN_PLATE_PIXELS:
            return False, {
                "reason": "plate_too_small_px",
                "plate_pixels": plate_pixels,
                "plate_area_ratio": plate_area_ratio,
                "min_plate_pixels": self.MIN_PLATE_PIXELS,
            }

        if plate_area_ratio < self.MIN_PLATE_AREA_RATIO:
            return False, {
                "reason": "plate_too_small_area",
                "plate_pixels": plate_pixels,
                "plate_area_ratio": plate_area_ratio,
                "min_plate_area_ratio": self.MIN_PLATE_AREA_RATIO,
            }

        area_mask = (plate | lo)
        ys, xs = np.where(area_mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        top_margin = y0
        left_margin = x0
        bottom_margin = (H - 1) - y1
        right_margin = (W - 1) - x1
        min_margin = int(min(top_margin, left_margin, bottom_margin, right_margin))

        severe_thr = int(self.CROP_SEVERE_MARGIN_RATIO * min(H, W))
        warn_thr = int(self.CROP_WARN_MARGIN_RATIO * min(H, W))

        if min_margin <= severe_thr:
            return False, {
                "reason": "cropped_severe",
                "bbox": [x0, y0, x1, y1],
                "min_margin": min_margin,
                "severe_thr": severe_thr,
                "warn_thr": warn_thr,
                "plate_area_ratio": plate_area_ratio,
                "plate_pixels": plate_pixels,
            }

        total = int(area_mask.sum())
        total_area_ratio = total / float(img_area + 1e-6)

        if min_margin <= warn_thr:
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

    def _clean_leftover_mask(self, pred: np.ndarray) -> np.ndarray:
        plate = (pred == self.PLATE)
        lo = (pred == self.LEFTOVER)

        try:
            import cv2  # type: ignore

            lo_u8 = lo.astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lo_u8, connectivity=8)

            cleaned = np.zeros_like(lo_u8)
            plate_area = int(plate.sum())
            min_area = max(20, int(self.MIN_LEFTOVER_AREA_RATIO * plate_area))

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    cleaned[labels == i] = 1

            return cleaned.astype(bool)
        except Exception:
            return lo

    def _weighted_leftover_ratio(self, pred: np.ndarray) -> float:
        plate = (pred == self.PLATE)
        lo = self._clean_leftover_mask(pred)

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

    def _empty_ratio(self, pred: np.ndarray) -> float:
        if self.USE_WEIGHTED:
            ratio = 1.0 - float(self._weighted_leftover_ratio(pred))
        else:
            leftover_pixels = int((pred == self.LEFTOVER).sum())
            plate_pixels = int((pred == self.PLATE).sum())
            total = leftover_pixels + plate_pixels
            ratio = 0.0 if total == 0 else (1.0 - (leftover_pixels / float(total)))

        ratio = max(0.0, min(1.0, ratio))

        if ratio >= self.EMPTY_RATIO_SNAP:
            ratio = 1.0

        return ratio

    def calculate_leftover_ratio(self, image: Image.Image) -> float:
        """호환성 때문에 함수명은 유지하지만, 실제 반환값은 '비운 비율(0~1)'"""
        img_np = np.array(image.convert("RGB"))
        result = inference_model(self.model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)

        ok, diag = self._shot_quality_gate(pred)
        if not ok:
            print("gate failed in calculate_leftover_ratio:", diag)

        return self._empty_ratio(pred)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        img_np = np.array(image.convert("RGB"))
        result = inference_model(self.model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)

        ok, diag = self._shot_quality_gate(pred)
        reason = diag.get("reason", "unknown")

        ratio = self._empty_ratio(pred)

        if not ok and reason == "no_plate":
            return {
                "status": "RETAKE",
                "empty_ratio": None,
                "message": "그릇(식판)을 화면 중앙에 두고 다시 촬영해주세요.",
                "diag": diag,
            }

        warn_msgs = {
            "plate_too_small_px": "그릇(식판) 인식이 작게 잡혔어요. 결과 정확도가 낮을 수 있어요.",
            "plate_too_small_area": "그릇(식판) 인식이 작게 잡혔어요. 결과 정확도가 낮을 수 있어요.",
            "cropped_severe": "식판이 화면 가장자리에 가까워 보여요. 다음엔 여백을 조금만 더 주세요.",
            "cropped_warn": "식판이 화면에 조금 타이트해요. 다음엔 여백을 조금만 더 두면 더 정확해요.",
        }

        msg = warn_msgs.get(reason)

        resp = {
            "status": "OK",
            "empty_ratio": float(ratio),
            "diag": diag,
        }

        # 기존 코드와의 호환이 필요하면 같이 넣기
        resp["leftover_ratio"] = float(ratio)

        if msg:
            resp["message"] = msg
            resp["low_confidence"] = True

        return resp