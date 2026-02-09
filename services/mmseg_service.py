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
        self.MIN_PLATE_AREA_RATIO = 0.25   # plate가 이미지 25% 미만이면 FAIL (너무 멀거나 식판 아님)
        self.MIN_PLATE_PIXELS = 5000       # 해상도에 따라 조절

        self.BORDER = 6                    # border strip 폭(px)
        self.MAX_SIDE_TOUCH_THR = 0.35     # 한 변에 plate가 너무 길게 붙으면(크롭) FAIL (0.35~0.45 튜닝)

        # 전체 plate 중 border에 걸린 비율이 너무 높고 + side touch도 높은 경우
        self.TOUCH_RATIO_THR = 0.08

        # 양념 번짐 눌러주는 weighted ratio
        self.USE_WEIGHTED = True
        self.DIST_TAU = 12                 # leftover 두께 기준(px) (해상도 따라 조절)
        self.W_EPS = 0.08                  # 번짐도 아주 조금은 잔반으로 치고 싶으면 0.05~0.15

    # --------------------------
    # A) 부분샷 
    # --------------------------
    def _shot_quality_gate(self, pred: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns:
            (ok, diag)
            ok=False면 "부분샷/무효샷"으로 보고 재촬영 유도
        """
        H, W = pred.shape
        img_area = H * W

        plate = (pred == self.PLATE)
        lo = (pred == self.LEFTOVER)

        plate_pixels = int(plate.sum())
        plate_area_ratio = plate_pixels / float(img_area + 1e-6)

        # 1) 완전 미검출 / plate 너무 작음
        if plate_pixels == 0:
            return False, {"reason": "no_plate", "plate_pixels": 0}
        if plate_pixels < self.MIN_PLATE_PIXELS:
            return False, {"reason": "plate_too_small_px", "plate_pixels": plate_pixels}
        if plate_area_ratio < self.MIN_PLATE_AREA_RATIO:
            return False, {"reason": "plate_too_small_area", "plate_area_ratio": plate_area_ratio}

        # 2) border touch 기반 "크롭" 판정
        b = int(self.BORDER)

        top = int(plate[:b, :].sum())
        bottom = int(plate[-b:, :].sum())
        left = int(plate[:, :b].sum())
        right = int(plate[:, -b:].sum())

        # 변별 접촉(해당 border strip 면적 대비 plate 비율)
        top_r = top / float(b * W + 1e-6)
        bottom_r = bottom / float(b * W + 1e-6)
        left_r = left / float(b * H + 1e-6)
        right_r = right / float(b * H + 1e-6)
        max_side = float(max(top_r, bottom_r, left_r, right_r))

        # 전체 plate 중 border에 걸친 비율(보조 신호)
        touch = top + bottom + left + right
        touch_ratio = touch / float(plate_pixels + 1e-6)

        # FAIL 조건 : 한쪽 변에 plate가 과하게 붙음
        if max_side > self.MAX_SIDE_TOUCH_THR:
            return False, {
                "reason": "cropped_by_side_touch",
                "plate_area_ratio": plate_area_ratio,
                "max_side_touch": max_side,
                "touch_ratio": touch_ratio,
            }

        # border에 너무 많이 걸리고 + side touch도 어느 정도 있음 이면 FAIL
        if touch_ratio > self.TOUCH_RATIO_THR and max_side > (self.MAX_SIDE_TOUCH_THR * 0.8):
            return False, {
                "reason": "cropped_by_touch_ratio",
                "plate_area_ratio": plate_area_ratio,
                "max_side_touch": max_side,
                "touch_ratio": touch_ratio,
            }

        # PASS
        total = int((plate | lo).sum())
        total_area_ratio = total / float(img_area + 1e-6)

        return True, {
            "reason": "ok",
            "plate_area_ratio": plate_area_ratio,
            "max_side_touch": max_side,
            "touch_ratio": touch_ratio,
            "total_area_ratio": total_area_ratio,
            "plate_pixels": plate_pixels,
        }


    # --------------------------
    # B) 양념 번짐 눌러주는 가중 잔반율
    # --------------------------
    def _weighted_leftover_ratio(self, img_rgb: np.ndarray, pred: np.ndarray) -> float:
        plate = (pred == self.PLATE)
        lo = (pred == self.LEFTOVER)

        denom = int((plate | lo).sum())
        if denom == 0:
            return 0.0

        # distance transform 기반 "두께" 가중치 (cv2 있으면 좋음)
        try:
            import cv2  # type: ignore
            lo_u8 = lo.astype(np.uint8)
            dist = cv2.distanceTransform(lo_u8, distanceType=cv2.DIST_L2, maskSize=3)
            w_thin = np.clip(dist / float(self.DIST_TAU), 0.0, 1.0)
        except Exception:
            # cv2 없으면 fallback: 가중치 없이 1로
            w_thin = np.ones_like(pred, dtype=np.float32)

        w = np.maximum(self.W_EPS, w_thin).astype(np.float32)
        numer = float((w * lo).sum())
        return numer / float(denom)
    
    
    def calculate_leftover_ratio(self, image: Image.Image) -> float:
        """
        Returns:
            잔반 비율 (0.0 ~ 1.0)
            = leftover_pixels / (plate_pixels + leftover_pixels)
        """
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]
        img_area = h * w

        result = inference_model(self.model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)  # (H, W)

        ok, _diag = self._shot_quality_gate(pred)
        if not ok:
            return 0.0

        if self.USE_WEIGHTED:
            return float(self._weighted_leftover_ratio(img_np, pred))

        leftover_pixels = int((pred == self.LEFTOVER).sum())
        plate_pixels = int((pred == self.PLATE).sum())
        total = leftover_pixels + plate_pixels
        if total == 0:
            return 0.0
        return leftover_pixels / float(total)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        img_np = np.array(image.convert("RGB"))
        result = inference_model(self.model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)

        ok, diag = self._shot_quality_gate(pred)
        if not ok:
            return {
                "status": "RETAKE",
                "leftover_ratio": None,
                "message": "식판 전체가 나오게 다시 촬영해주세요.",
                "diag": diag,
            }

        # OK면 ratio 계산
        if self.USE_WEIGHTED:
            ratio = float(self._weighted_leftover_ratio(img_np, pred))
        else:
            leftover_pixels = int((pred == self.LEFTOVER).sum())
            plate_pixels = int((pred == self.PLATE).sum())
            total = leftover_pixels + plate_pixels
            ratio = 0.0 if total == 0 else (leftover_pixels / float(total))

        return {"status": "OK", "leftover_ratio": float(ratio), "diag": diag}