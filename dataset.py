import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter


class ShanghaiTechPartADataset(Dataset):
    """
    PyTorch Dataset for ShanghaiTech Part A crowd counting.
    Loads images and their corresponding .mat annotations,
    converts head points into Gaussian density maps,
    and returns (image_tensor, density_map_tensor).
    """

    def __init__(self, img_dir, gt_dir, img_size=(512, 512), density_scale=8):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.density_scale = density_scale

        self.image_names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.image_names)

    def _resolve_gt_path(self, img_name):
        """Find the ground-truth .mat file for the given image."""
        base = os.path.splitext(img_name)[0]
        candidates = [
            os.path.join(self.gt_dir, f"GT_{base}.mat"),
            os.path.join(self.gt_dir, f"{base}.mat"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        # Fallback: search by matching the base name inside the gt directory
        for file_name in sorted(os.listdir(self.gt_dir)):
            if file_name.lower().endswith(".mat"):
                if base.lower() in file_name.lower():
                    return os.path.join(self.gt_dir, file_name)

        raise FileNotFoundError(f"Ground truth not found for {img_name}")

    def _coerce_points(self, data):
        """Convert annotation payloads into a (N, 2) array of point coordinates."""
        if data is None:
            return None

        arr = np.asarray(data)
        if arr.size == 0:
            return None

        if arr.ndim == 1:
            if arr.size >= 2 and np.issubdtype(arr.dtype, np.number):
                return arr.reshape(1, 2).astype(np.float32)
            return None

        if arr.ndim >= 2 and arr.shape[-1] == 2:
            return arr.astype(np.float32).reshape(-1, 2)

        if arr.dtype == object:
            flat_points = []
            for item in arr.reshape(-1):
                if isinstance(item, np.ndarray):
                    if item.size >= 2:
                        flat_points.append(item.reshape(-1)[:2])
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    flat_points.append(np.asarray(item[:2], dtype=np.float32))
            if flat_points:
                return np.stack(flat_points, axis=0).astype(np.float32)

        return None

    def _load_points(self, gt_path):
        """Load head points from either annPoints or image_info annotations."""
        mat = sio.loadmat(gt_path)

        # Case 1: direct point list
        for key in ["annPoints", "points", "points_xy"]:
            if key in mat:
                pts = self._coerce_points(mat[key])
                if pts is not None:
                    return pts

        # Case 2: ShanghaiTech-style image_info structure
        if "image_info" in mat:
            info = mat["image_info"]

            # Structured array style: info[0,0]['location']
            if hasattr(info, "dtype") and info.dtype.names is not None and "location" in info.dtype.names:
                location = info[0, 0]["location"]
                pts = self._coerce_points(location)
                if pts is not None:
                    return pts

            # Nested array style used by some ShanghaiTech files
            candidates = []
            try:
                candidates.append(info[0, 0][0, 0])
            except Exception:
                pass
            try:
                candidates.append(info[0, 0][0][0][0])
            except Exception:
                pass
            try:
                candidates.append(info[0, 0][0][0])
            except Exception:
                pass
            try:
                candidates.append(info[0, 0])
            except Exception:
                pass

            for candidate in candidates:
                pts = self._coerce_points(candidate)
                if pts is not None:
                    return pts

        raise ValueError(f"Could not find valid head-point annotations in {gt_path}")

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = self._resolve_gt_path(img_name)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))

        h, w = self.img_size
        density = np.zeros((h, w), dtype=np.float32)

        points = self._load_points(gt_path)
        if points is not None:
            scale_x = w / orig_w
            scale_y = h / orig_h

            for p in points:
                x = int(round(float(p[0]) * scale_x))
                y = int(round(float(p[1]) * scale_y))
                if 0 <= x < w and 0 <= y < h:
                    density[y, x] += 1.0

        density = gaussian_filter(density, sigma=4)

        ds_h, ds_w = h // self.density_scale, w // self.density_scale
        density = cv2.resize(density, (ds_w, ds_h))
        density = density * (self.density_scale ** 2)

        img_tensor = self.transform(img)
        density_tensor = torch.from_numpy(density).float().unsqueeze(0)

        return img_tensor, density_tensor