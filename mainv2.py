#!/usr/bin/env python
import os
import sys
import math
import argparse
import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import optuna
import csv

try:
    import timm  # noqa: F401
except Exception:
    timm = None


# ---------------------------
# Utils & data loading
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_zip_if_exists(zip_name: str, extract_to: str = "."):
    if os.path.exists(zip_name):
        print(f"[INFO] Extracting {zip_name} to {extract_to} ...")
        import zipfile
        with zipfile.ZipFile(zip_name, "r") as zf:
            zf.extractall(extract_to)
    else:
        print(f"[WARN] {zip_name} not found, assuming already extracted.")


def find_dataset_root_for_classes(root: str, required_subdirs: List[str]) -> Optional[str]:
    """Find a directory under root that contains all required_subdirs."""
    required_lower = set([d.lower() for d in required_subdirs])
    for dirpath, dirnames, _ in os.walk(root):
        dirnames_lower = set([d.lower() for d in dirnames])
        if required_lower.issubset(dirnames_lower):
            print(f"[INFO] Found dataset root for {required_subdirs} at: {dirpath}")
            return dirpath
    return None


# ---------- robust NPY loading (with allow_pickle=True) ----------

def _extract_2d_numeric_array(obj: Any) -> Optional[np.ndarray]:
    """
    Try to robustly extract a 2D numeric array from weird nested NPY contents.
    """
    if isinstance(obj, np.ndarray):
        # Direct 2D numeric
        if obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
            return obj.astype(np.float32)

        # 3D numeric (channels or stacked)
        if obj.ndim == 3 and np.issubdtype(obj.dtype, np.number):
            arr = obj.astype(np.float32)
            if arr.shape[0] in (1, 3):
                # assume channels-first
                return arr.mean(axis=0)
            if arr.shape[-1] in (1, 3):
                # assume channels-last
                return arr.mean(axis=-1)
            # fallback: try flatten to square
            flat = arr[0].ravel()
            side = int(math.sqrt(flat.size))
            if side * side == flat.size:
                return flat[: side * side].reshape(side, side)
            return None

        # Object array: recurse
        if obj.dtype == object:
            if obj.ndim == 0:
                return _extract_2d_numeric_array(obj.item())
            if obj.ndim == 1:
                for el in obj:
                    arr = _extract_2d_numeric_array(el)
                    if arr is not None:
                        return arr
            it = np.nditer(obj, flags=["refs_ok", "multi_index"], op_flags=["readonly"])
            for x in it:
                arr = _extract_2d_numeric_array(x.item())
                if arr is not None:
                    return arr
            return None

        # 1D numeric: attempt to reshape to square
        if obj.ndim == 1 and np.issubdtype(obj.dtype, np.number):
            side = int(math.sqrt(obj.size))
            if side * side == obj.size:
                return obj.astype(np.float32).reshape(side, side)
            return None

        # Higher-dim numeric: peel off dims until 2D
        if np.issubdtype(obj.dtype, np.number):
            if obj.ndim >= 2:
                img = obj
                while img.ndim > 2:
                    img = img[0]
                if img.ndim == 2:
                    return img.astype(np.float32)
            return None

    if isinstance(obj, dict):
        for key in ["image", "img", "data", "map", "arr"]:
            if key in obj:
                arr = _extract_2d_numeric_array(obj[key])
                if arr is not None:
                    return arr
        for v in obj.values():
            arr = _extract_2d_numeric_array(v)
            if arr is not None:
                return arr
        return None

    if isinstance(obj, (list, tuple)):
        for el in obj:
            arr = _extract_2d_numeric_array(el)
            if arr is not None:
                return arr
        return None

    return None


def robust_load_npy(path: str,
                    target_size: int = 64,
                    device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Safe loader: allow_pickle=True, handles weird objects, and replaces
    failures with random noise.
    """
    try:
        raw = np.load(path, allow_pickle=True)
        arr = _extract_2d_numeric_array(raw)
        if arr is None or arr.size == 0:
            raise ValueError("Could not extract 2D numeric image from object")

        img = torch.from_numpy(arr.astype(np.float32))
        # normalize to [0, 1]
        min_val = torch.min(img)
        img = img - min_val
        max_val = torch.max(img)
        if max_val > 0:
            img = img / max_val
        else:
            img = torch.zeros_like(img)

        # to (1, H, W)
        img = img.unsqueeze(0)
        # force-resize to 64x64 via bicubic
        img = img.unsqueeze(0)
        img = F.interpolate(
            img,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )
        img = img.squeeze(0)
        return img
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}. Using random noise.")
        noise = torch.rand(1, target_size, target_size)
        return noise


# ---------------------------
# Datasets
# ---------------------------

class NoSubDataset(Dataset):
    """For MAE pre-training (no_sub class only)."""

    def __init__(self, file_paths: List[str], target_size: int = 64):
        self.file_paths = file_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = robust_load_npy(path, target_size=self.target_size)
        return img


class ClassificationDataset(Dataset):
    """For 3-way classification (axion, cdm, no_sub)."""

    def __init__(self,
                 samples: List[Tuple[str, int]],
                 num_classes: int,
                 target_size: int = 64):
        self.samples = samples
        self.num_classes = num_classes
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = robust_load_npy(path, target_size=self.target_size)
        return img, label


class SuperResolutionDataset(Dataset):
    """For LR→HR super-resolution (Task VI B)."""

    def __init__(self,
                 pairs: List[Tuple[str, str]],
                 target_size: int = 64):
        self.pairs = pairs
        self.target_size = target_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        lr_img = robust_load_npy(lr_path, target_size=self.target_size)
        hr_img = robust_load_npy(hr_path, target_size=self.target_size)
        return lr_img, hr_img


def build_task_VI_A_datasets(dataset1_root: str,
                             val_fraction: float = 0.1,
                             target_size: int = 64,
                             seed: int = 42):
    """
    Build:
      - MAE pretrain dataset (no_sub only)
      - Classification train/val datasets (3-way)
    """
    classes = ["axion", "cdm", "no_sub"]
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    all_samples: List[Tuple[str, int]] = []
    no_sub_paths: List[str] = []

    for cls in classes:
        cls_dir = os.path.join(dataset1_root, cls)
        if not os.path.isdir(cls_dir):
            raise RuntimeError(f"Expected class dir {cls_dir} not found")
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(cls_dir, fname)
            label = class_to_idx[cls]
            all_samples.append((fpath, label))
            if cls == "no_sub":
                no_sub_paths.append(fpath)

    print(f"[INFO] Total samples: {len(all_samples)} "
          f"(axion/cdm/no_sub: "
          f"{len([s for s in all_samples if s[1]==class_to_idx['axion']])}/"
          f"{len([s for s in all_samples if s[1]==class_to_idx['cdm']])}/"
          f"{len(no_sub_paths)})")

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    val_size = int(len(all_samples) * val_fraction)
    train_samples = all_samples[:-val_size]
    val_samples = all_samples[-val_size:]

    train_ds = ClassificationDataset(train_samples,
                                     num_classes=len(classes),
                                     target_size=target_size)
    val_ds = ClassificationDataset(val_samples,
                                   num_classes=len(classes),
                                   target_size=target_size)
    mae_ds = NoSubDataset(no_sub_paths, target_size=target_size)

    return mae_ds, train_ds, val_ds, class_to_idx


def build_task_VI_B_datasets(dataset2_root: str,
                             val_fraction: float = 0.1,
                             target_size: int = 64,
                             seed: int = 42):
    """
    Build SR train/val datasets from HR/LR folders.
    """
    hr_dir = os.path.join(dataset2_root, "HR")
    lr_dir = os.path.join(dataset2_root, "LR")

    if not os.path.isdir(hr_dir) or not os.path.isdir(lr_dir):
        raise RuntimeError(f"HR or LR directory not found under {dataset2_root}")

    hr_files = {
        os.path.basename(f): os.path.join(hr_dir, f)
        for f in os.listdir(hr_dir) if f.endswith(".npy")
    }
    lr_files = {
        os.path.basename(f): os.path.join(lr_dir, f)
        for f in os.listdir(lr_dir) if f.endswith(".npy")
    }

    common_names = sorted(list(set(hr_files.keys()).intersection(lr_files.keys())))
    pairs = [(lr_files[name], hr_files[name]) for name in common_names]

    print(f"[INFO] Super-resolution pairs (LR/HR): {len(pairs)}")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    val_size = int(len(pairs) * val_fraction)
    train_pairs = pairs[:-val_size]
    val_pairs = pairs[-val_size:]

    train_ds = SuperResolutionDataset(train_pairs, target_size=target_size)
    val_ds = SuperResolutionDataset(val_pairs, target_size=target_size)
    return train_ds, val_ds


# ---------------------------
# ViT + MAE
# ---------------------------

class PatchEmbed(nn.Module):
    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 4,
                 in_chans: int = 1,
                 embed_dim: int = 192):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                          batch_first=True,
                                          dropout=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim,
                       hidden_features=int(dim * mlp_ratio),
                       drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 4,
                 in_chans: int = 1,
                 embed_dim: int = 192,
                 depth: int = 6,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             drop=drop_rate,
                             attn_drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features_from_patches(self, x_patches: torch.Tensor) -> torch.Tensor:
        B, N, C = x_patches.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_patches), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x_img)
        return self.forward_features_from_patches(patches)


class MaskedAutoencoderViT(nn.Module):
    def __init__(self,
                 encoder: ViTEncoder,
                 img_size: int = 64,
                 patch_size: int = 4,
                 in_chans: int = 1,
                 mask_ratio: float = 0.75,
                 decoder_dim: int = 192,
                 decoder_depth: int = 2,
                 decoder_num_heads: int = 3):
        super().__init__()
        self.encoder = encoder
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio

        self.num_patches = encoder.patch_embed.num_patches
        self.decoder_embed = nn.Linear(encoder.embed_dim, decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dim=decoder_dim,
                             num_heads=decoder_num_heads,
                             mlp_ratio=4.0,
                             drop=0.0,
                             attn_drop=0.0)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)

    def random_mask(self, x_patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x_patches.shape
        num_mask = int(N * self.mask_ratio)
        mask = torch.zeros(B, N, dtype=torch.bool, device=x_patches.device)
        for i in range(B):
            perm = torch.randperm(N, device=x_patches.device)
            mask_idx = perm[:num_mask]
            mask[i, mask_idx] = True

        x_masked = x_patches.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L = x.shape
        h = w = int(math.sqrt(N))
        assert h * w == N, "Number of patches must be a square"
        x = x.reshape(B, h, w, self.patch_size, self.patch_size, self.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, self.in_chans, h * self.patch_size, w * self.patch_size)
        return x

    def forward(self, x_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = self.encoder.patch_embed(x_img)
        patches_masked, mask = self.random_mask(patches)
        encoded = self.encoder.forward_features_from_patches(patches_masked)
        encoded_patches = encoded[:, 1:, :]
        dec = self.decoder_embed(encoded_patches)
        for blk in self.decoder_blocks:
            dec = blk(dec)
        dec = self.decoder_norm(dec)
        pred = self.decoder_pred(dec)
        recon = self.unpatchify(pred)
        return recon, mask


# ---------------------------
# Downstream heads
# ---------------------------

class ViTClassifier(nn.Module):
    def __init__(self, encoder: ViTEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(x)
        cls_feat = tokens[:, 0]
        logits = self.head(cls_feat)
        return logits, cls_feat


class SRHead(nn.Module):
    """
    SR upsampling head using bilinear upsampling + conv (no PixelShuffle).
    Input feature map: (B, C, 16, 16) → output: (B, 1, 64, 64)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(embed_dim // 4, 1, kernel_size=3, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.final_conv(x)
        x = self.act(x)
        return x


class ViTSuperResolution(nn.Module):
    def __init__(self, encoder: ViTEncoder):
        super().__init__()
        self.encoder = encoder
        self.sr_head = SRHead(embed_dim=encoder.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)
        patch_tokens = tokens[:, 1:, :]
        B, N, C = patch_tokens.shape
        h = w = int(math.sqrt(N))
        feat = patch_tokens.transpose(1, 2).reshape(B, C, h, w)
        out = self.sr_head(feat)
        return out


# ---------------------------
# Training / Evaluation
# ---------------------------

def train_mae(model: MaskedAutoencoderViT,
              dataloader: DataLoader,
              device: torch.device,
              epochs: int = 10,
              lr: float = 1e-4) -> float:
    """
    Train MAE, return final epoch loss for logging / mask-ratio ablations.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    last_epoch_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"MAE Epoch {epoch+1}/{epochs}")
        for imgs in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})
        epoch_loss = running_loss / len(dataloader.dataset)
        last_epoch_loss = epoch_loss
        print(f"[MAE] Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.6f}")
    return last_epoch_loss


def evaluate_classifier(model: ViTClassifier,
                        dataloader: DataLoader,
                        device: torch.device,
                        num_classes: int) -> Dict:
    """
    Returns:
      - auc_macro, fpr, tpr
      - probs, labels, features
      - accuracy, f1_macro, y_pred
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_feats = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, feats = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_feats.append(feats.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_feats = torch.cat(all_feats, dim=0)

    probs = torch.softmax(all_logits, dim=1).numpy()
    y_true = all_labels.numpy()
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    try:
        auc_macro = roc_auc_score(y_true_bin, probs,
                                  average="macro",
                                  multi_class="ovr")
    except ValueError as e:
        print(f"[WARN] AUC computation error: {e}. Using NaN.")
        auc_macro = float("nan")

    fpr_dict = {}
    tpr_dict = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr

    return {
        "auc_macro": auc_macro,
        "fpr": fpr_dict,
        "tpr": tpr_dict,
        "probs": probs,
        "labels": y_true,
        "features": all_feats.numpy(),
        "accuracy": acc,
        "f1_macro": f1,
        "y_pred": y_pred,
    }


def plot_roc_curves(metrics: Dict,
                    class_names: List[str],
                    save_path: str = "roc_curve.png"):
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        fpr = metrics["fpr"][i]
        tpr = metrics["tpr"][i]
        plt.plot(fpr, tpr, label=f"Class {cls}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] ROC curve saved to {save_path}")


def plot_tsne(features: np.ndarray,
              labels: np.ndarray,
              class_names: List[str],
              save_path: str = "tsne.png",
              max_points: int = 2000):
    if features.shape[0] > max_points:
        idx = np.random.choice(features.shape[0], max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    feats_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        plt.scatter(feats_2d[mask, 0],
                    feats_2d[mask, 1],
                    s=5,
                    label=cls_name,
                    alpha=0.7)
    plt.legend()
    plt.title("t-SNE of ViT Encoder Latent Space")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] t-SNE plot saved to {save_path}")


def evaluate_sr(model: ViTSuperResolution,
                dataloader: DataLoader,
                device: torch.device) -> Dict[str, float]:
    model.eval()
    mse_vals = []
    psnr_vals = []
    ssim_vals = []

    with torch.no_grad():
        for lr_img, hr_img in dataloader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            pred = model(lr_img)

            mse_batch = F.mse_loss(pred, hr_img, reduction="none")
            mse_batch = mse_batch.mean(dim=[1, 2, 3]).cpu().numpy()

            for i in range(pred.size(0)):
                pred_np = pred[i, 0].cpu().numpy()
                hr_np = hr_img[i, 0].cpu().numpy()

                mse_vals.append(mse_batch[i])
                psnr_vals.append(
                    peak_signal_noise_ratio(hr_np, pred_np, data_range=1.0)
                )
                ssim_vals.append(
                    structural_similarity(hr_np, pred_np, data_range=1.0)
                )

    metrics = {
        "mse": float(np.mean(mse_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
    }
    print(f"[SR] MSE: {metrics['mse']:.6f}, "
          f"PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}")
    return metrics


def visualize_sr_example(model: ViTSuperResolution,
                         dataloader: DataLoader,
                         device: torch.device,
                         save_path: str = "sr_example.png"):
    model.eval()
    with torch.no_grad():
        for lr_img, hr_img in dataloader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            pred = model(lr_img)
            lr_np = lr_img[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            hr_np = hr_img[0, 0].cpu().numpy()
            break

    plt.figure(figsize=(12, 4))
    vmin, vmax = 0.0, 1.0

    plt.subplot(1, 3, 1)
    plt.title("Input LR")
    plt.imshow(lr_np, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Model Prediction")
    plt.imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth HR")
    plt.imshow(hr_np, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] SR comparison image saved to {save_path}")


def visualize_sr_grid(model: ViTSuperResolution,
                      dataloader: DataLoader,
                      device: torch.device,
                      n_examples: int = 4,
                      save_path: str = "sr_grid.png"):
    """Grid with multiple LR / Pred / HR triplets for sanity-check."""
    model.eval()
    lr_list = []
    pred_list = []
    hr_list = []
    with torch.no_grad():
        for lr_img, hr_img in dataloader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            pred = model(lr_img)
            for i in range(lr_img.size(0)):
                lr_list.append(lr_img[i, 0].cpu().numpy())
                pred_list.append(pred[i, 0].cpu().numpy())
                hr_list.append(hr_img[i, 0].cpu().numpy())
                if len(lr_list) >= n_examples:
                    break
            if len(lr_list) >= n_examples:
                break

    cols = 3
    rows = n_examples
    plt.figure(figsize=(cols * 3.5, rows * 3.0))
    vmin, vmax = 0.0, 1.0
    for i in range(n_examples):
        plt.subplot(rows, cols, i * cols + 1)
        plt.imshow(lr_list[i], cmap="viridis", vmin=vmin, vmax=vmax)
        if i == 0:
            plt.title("Input LR")
        plt.axis("off")

        plt.subplot(rows, cols, i * cols + 2)
        plt.imshow(pred_list[i], cmap="viridis", vmin=vmin, vmax=vmax)
        if i == 0:
            plt.title("Prediction")
        plt.axis("off")

        plt.subplot(rows, cols, i * cols + 3)
        plt.imshow(hr_list[i], cmap="viridis", vmin=vmin, vmax=vmax)
        if i == 0:
            plt.title("Ground Truth HR")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] SR grid saved to {save_path}")


# ---------------------------
# Extra sanity checks: confusion & calibration
# ---------------------------

def plot_confusion_and_report(metrics: Dict,
                              class_names: List[str],
                              prefix: str = "outputs/cls_pretrained"):
    y_true = metrics["labels"]
    y_pred = metrics["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = f"{prefix}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   digits=4)
    rep_path = f"{prefix}_classification_report.txt"
    with open(rep_path, "w") as f:
        f.write(report)
    print(f"[INFO] Classification report saved to {rep_path}")
    print(report)


def plot_reliability_diagram(metrics: Dict,
                             save_path: str = "outputs/reliability_pretrained.png",
                             n_bins: int = 10):
    """
    Reliability diagram for top-1 predictions:
      x-axis = predicted confidence (max prob)
      y-axis = observed accuracy in that bin.
    """
    probs = metrics["probs"]
    labels = metrics["labels"]
    y_pred = metrics["y_pred"]

    confidences = probs.max(axis=1)
    correctness = (y_pred == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    acc_in_bin = []
    conf_in_bin = []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if mask.sum() > 0:
            acc_in_bin.append(correctness[mask].mean())
            conf_in_bin.append(confidences[mask].mean())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if len(conf_in_bin) > 0:
        plt.plot(conf_in_bin, acc_in_bin, marker="o", label="Model")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Reliability Diagram (Top-1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Reliability diagram saved to {save_path}")


# ---------------------------
# Optuna objectives
# ---------------------------

def objective_classification(trial: optuna.Trial,
                             train_ds: ClassificationDataset,
                             val_ds: ClassificationDataset,
                             mae_encoder_state: Dict,
                             num_classes: int,
                             device: torch.device,
                             batch_size: int = 64) -> float:
    lr = trial.suggest_loguniform("lr", 1e-5, 5e-4)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    drop_rate = trial.suggest_float("drop_rate", 0.0, 0.3)

    encoder = ViTEncoder(img_size=64,
                         patch_size=4,
                         in_chans=1,
                         embed_dim=192,
                         depth=6,
                         num_heads=3,
                         mlp_ratio=4.0,
                         drop_rate=drop_rate)
    encoder.load_state_dict(mae_encoder_state)
    model = ViTClassifier(encoder, num_classes=num_classes).to(device)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    max_epochs = 5
    for epoch in range(max_epochs):
        model.train()
        pbar = tqdm(train_loader,
                    desc=f"[Optuna CLS] Epoch {epoch+1}/{max_epochs}",
                    leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    metrics = evaluate_classifier(model, val_loader, device, num_classes)
    auc_macro = metrics["auc_macro"]
    print(f"[Optuna CLS] Trial {trial.number} AUC: {auc_macro:.4f}")
    return auc_macro


def objective_sr(trial: optuna.Trial,
                 train_ds: SuperResolutionDataset,
                 val_ds: SuperResolutionDataset,
                 mae_encoder_state: Dict,
                 device: torch.device,
                 batch_size: int = 64) -> float:
    lr = trial.suggest_loguniform("lr", 1e-5, 5e-4)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    drop_rate = trial.suggest_float("drop_rate", 0.0, 0.3)

    encoder = ViTEncoder(img_size=64,
                         patch_size=4,
                         in_chans=1,
                         embed_dim=192,
                         depth=6,
                         num_heads=3,
                         mlp_ratio=4.0,
                         drop_rate=drop_rate)
    encoder.load_state_dict(mae_encoder_state)
    model = ViTSuperResolution(encoder).to(device)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.MSELoss()

    max_epochs = 5
    for epoch in range(max_epochs):
        model.train()
        pbar = tqdm(train_loader,
                    desc=f"[Optuna SR] Epoch {epoch+1}/{max_epochs}",
                    leave=False)
        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            optimizer.zero_grad()
            pred = model(lr_img)
            loss = criterion(pred, hr_img)
            loss.backward()
            optimizer.step()

    metrics = evaluate_sr(model, val_loader, device)
    psnr = metrics["psnr"]
    print(f"[Optuna SR] Trial {trial.number} PSNR: {psnr:.4f}")
    return psnr


# ---------------------------
# Ablation experiments
# ---------------------------

def classifier_experiment(name: str,
                          use_pretrained: bool,
                          freeze_encoder: bool,
                          mae_encoder_state: Dict,
                          train_ds: ClassificationDataset,
                          val_ds: ClassificationDataset,
                          class_names: List[str],
                          device: torch.device,
                          batch_size: int,
                          epochs: int,
                          lr: float,
                          weight_decay: float,
                          drop_rate: float) -> Dict:
    print(f"[ABLATION CLS] Starting experiment '{name}' "
          f"(pretrained={use_pretrained}, freeze_encoder={freeze_encoder})")
    encoder = ViTEncoder(img_size=64,
                         patch_size=4,
                         in_chans=1,
                         embed_dim=192,
                         depth=6,
                         num_heads=3,
                         mlp_ratio=4.0,
                         drop_rate=drop_rate)
    if use_pretrained:
        encoder.load_state_dict(mae_encoder_state)

    model = ViTClassifier(encoder, num_classes=len(class_names)).to(device)

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"[{name}] CLS Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[{name}] CLS Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.6f}")

    metrics = evaluate_classifier(model, val_loader, device, num_classes=len(class_names))
    print(f"[ABLATION CLS][{name}] AUC={metrics['auc_macro']:.4f}, "
          f"ACC={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    return metrics


def sr_experiment(name: str,
                  use_pretrained: bool,
                  mae_encoder_state: Dict,
                  train_ds: SuperResolutionDataset,
                  val_ds: SuperResolutionDataset,
                  device: torch.device,
                  batch_size: int,
                  epochs: int,
                  lr: float,
                  weight_decay: float,
                  drop_rate: float) -> Dict[str, float]:
    print(f"[ABLATION SR] Starting experiment '{name}' "
          f"(pretrained={use_pretrained})")
    encoder = ViTEncoder(img_size=64,
                         patch_size=4,
                         in_chans=1,
                         embed_dim=192,
                         depth=6,
                         num_heads=3,
                         mlp_ratio=4.0,
                         drop_rate=drop_rate)
    if use_pretrained:
        encoder.load_state_dict(mae_encoder_state)

    model = ViTSuperResolution(encoder).to(device)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"[{name}] SR Epoch {epoch+1}/{epochs}")
        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            optimizer.zero_grad()
            pred = model(lr_img)
            loss = criterion(pred, hr_img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * lr_img.size(0)
            pbar.set_postfix({"loss": loss.item()})
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[{name}] SR Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.6f}")

    metrics = evaluate_sr(model, val_loader, device)
    print(f"[ABLATION SR][{name}] MSE={metrics['mse']:.6f}, "
          f"PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}")
    return metrics


def run_ablation_experiments(mae_encoder_state: Dict,
                             train_cls_ds: ClassificationDataset,
                             val_cls_ds: ClassificationDataset,
                             train_sr_ds: SuperResolutionDataset,
                             val_sr_ds: SuperResolutionDataset,
                             class_names: List[str],
                             device: torch.device,
                             batch_size: int,
                             epochs_cls: int,
                             epochs_sr: int,
                             cls_lr: float,
                             cls_wd: float,
                             cls_drop: float,
                             sr_lr: float,
                             sr_wd: float,
                             sr_drop: float):
    print("[INFO] === Running ablation experiments ===")

    # Classification ablations
    cls_results = []
    for name, use_pretrained, freeze_encoder in [
        ("pretrained_full", True, False),
        ("pretrained_frozen", True, True),
        ("scratch_full", False, False),
    ]:
        m = classifier_experiment(name,
                                  use_pretrained,
                                  freeze_encoder,
                                  mae_encoder_state,
                                  train_cls_ds,
                                  val_cls_ds,
                                  class_names,
                                  device,
                                  batch_size,
                                  epochs_cls,
                                  cls_lr,
                                  cls_wd,
                                  cls_drop)
        cls_results.append({
            "experiment": name,
            "auc_macro": m["auc_macro"],
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
        })

    cls_csv = "outputs/ablation_cls.csv"
    with open(cls_csv, "w", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=["experiment", "auc_macro", "accuracy", "f1_macro"])
        writer.writeheader()
        writer.writerows(cls_results)
    print(f"[INFO] Classification ablation results saved to {cls_csv}")

    # SR ablations
    sr_results = []
    for name, use_pretrained in [
        ("pretrained_sr", True),
        ("scratch_sr", False),
    ]:
        m = sr_experiment(name,
                          use_pretrained,
                          mae_encoder_state,
                          train_sr_ds,
                          val_sr_ds,
                          device,
                          batch_size,
                          epochs_sr,
                          sr_lr,
                          sr_wd,
                          sr_drop)
        sr_results.append({
            "experiment": name,
            "mse": m["mse"],
            "psnr": m["psnr"],
            "ssim": m["ssim"],
        })

    sr_csv = "outputs/ablation_sr.csv"
    with open(sr_csv, "w", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=["experiment", "mse", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(sr_results)
    print(f"[INFO] Super-resolution ablation results saved to {sr_csv}")


# ---------------------------
# MAE mask-ratio experiments
# ---------------------------

def mask_ratio_experiment(mask_ratio: float,
                          mae_loader: DataLoader,
                          train_cls_ds: ClassificationDataset,
                          val_cls_ds: ClassificationDataset,
                          train_sr_ds: SuperResolutionDataset,
                          val_sr_ds: SuperResolutionDataset,
                          class_names: List[str],
                          device: torch.device,
                          batch_size: int,
                          mae_epochs: int,
                          mae_lr: float,
                          cls_epochs: int,
                          cls_lr: float,
                          cls_wd: float,
                          cls_drop: float,
                          sr_epochs: int,
                          sr_lr: float,
                          sr_wd: float,
                          sr_drop: float) -> Dict[str, float]:
    """
    For a given MAE mask_ratio:
      - train MAE
      - fine-tune CLS and SR
      - return combined metrics
    """
    print(f"[MASK-ABLATION] Running mask_ratio={mask_ratio:.2f}")

    # --- MAE pretraining for this ratio ---
    encoder = ViTEncoder(img_size=64,
                         patch_size=4,
                         in_chans=1,
                         embed_dim=192,
                         depth=6,
                         num_heads=3,
                         mlp_ratio=4.0,
                         drop_rate=0.0)
    mae_model = MaskedAutoencoderViT(encoder=encoder,
                                     img_size=64,
                                     patch_size=4,
                                     in_chans=1,
                                     mask_ratio=mask_ratio,
                                     decoder_dim=192,
                                     decoder_depth=2,
                                     decoder_num_heads=3)
    mae_final_loss = train_mae(mae_model,
                               mae_loader,
                               device=device,
                               epochs=mae_epochs,
                               lr=mae_lr)
    mae_encoder_state = mae_model.encoder.state_dict()

    # --- CLS fine-tuning ---
    cls_metrics = classifier_experiment(
        name=f"mask_{mask_ratio:.2f}_cls",
        use_pretrained=True,
        freeze_encoder=False,
        mae_encoder_state=mae_encoder_state,
        train_ds=train_cls_ds,
        val_ds=val_cls_ds,
        class_names=class_names,
        device=device,
        batch_size=batch_size,
        epochs=cls_epochs,
        lr=cls_lr,
        weight_decay=cls_wd,
        drop_rate=cls_drop,
    )

    # --- SR fine-tuning ---
    sr_metrics = sr_experiment(
        name=f"mask_{mask_ratio:.2f}_sr",
        use_pretrained=True,
        mae_encoder_state=mae_encoder_state,
        train_ds=train_sr_ds,
        val_ds=val_sr_ds,
        device=device,
        batch_size=batch_size,
        epochs=sr_epochs,
        lr=sr_lr,
        weight_decay=sr_wd,
        drop_rate=sr_drop,
    )

    return {
        "mask_ratio": mask_ratio,
        "mae_final_loss": mae_final_loss,
        "cls_auc_macro": cls_metrics["auc_macro"],
        "cls_accuracy": cls_metrics["accuracy"],
        "cls_f1_macro": cls_metrics["f1_macro"],
        "sr_mse": sr_metrics["mse"],
        "sr_psnr": sr_metrics["psnr"],
        "sr_ssim": sr_metrics["ssim"],
    }


def run_mask_ratio_experiments(mask_ratios: List[float],
                               mae_loader: DataLoader,
                               train_cls_ds: ClassificationDataset,
                               val_cls_ds: ClassificationDataset,
                               train_sr_ds: SuperResolutionDataset,
                               val_sr_ds: SuperResolutionDataset,
                               class_names: List[str],
                               device: torch.device,
                               batch_size: int,
                               mae_epochs: int,
                               mae_lr: float,
                               cls_epochs: int,
                               cls_lr: float,
                               cls_wd: float,
                               cls_drop: float,
                               sr_epochs: int,
                               sr_lr: float,
                               sr_wd: float,
                               sr_drop: float):
    print("[INFO] === Running MAE mask-ratio experiments ===")
    results = []
    for r in mask_ratios:
        res = mask_ratio_experiment(
            mask_ratio=r,
            mae_loader=mae_loader,
            train_cls_ds=train_cls_ds,
            val_cls_ds=val_cls_ds,
            train_sr_ds=train_sr_ds,
            val_sr_ds=val_sr_ds,
            class_names=class_names,
            device=device,
            batch_size=batch_size,
            mae_epochs=mae_epochs,
            mae_lr=mae_lr,
            cls_epochs=cls_epochs,
            cls_lr=cls_lr,
            cls_wd=cls_wd,
            cls_drop=cls_drop,
            sr_epochs=sr_epochs,
            sr_lr=sr_lr,
            sr_wd=sr_wd,
            sr_drop=sr_drop,
        )
        results.append(res)

    csv_path = "outputs/mask_ratio_ablation.csv"
    fieldnames = [
        "mask_ratio",
        "mae_final_loss",
        "cls_auc_macro",
        "cls_accuracy",
        "cls_f1_macro",
        "sr_mse",
        "sr_psnr",
        "sr_ssim",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[INFO] Mask-ratio ablation results saved to {csv_path}")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DeepLense GSoC 2025 Task VI: MAE + Classification + Super-Resolution with Ablations"
    )
    parser.add_argument("--data_root",
                        type=str,
                        default=".",
                        help="Root directory containing Dataset1.zip and Dataset2.zip")
    parser.add_argument("--mae_epochs", type=int, default=10)
    parser.add_argument("--mae_lr", type=float, default=1e-4)
    parser.add_argument("--mae_mask_ratio", type=float, default=0.75,
                        help="Mask ratio for the main MAE pre-training run.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_optuna", action="store_true",
                        help="Use Optuna for hyperparameter tuning")
    parser.add_argument("--optuna_trials_cls", type=int, default=5)
    parser.add_argument("--optuna_trials_sr", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_ablation", action="store_true",
                        help="Run ablation experiments (scratch vs pretrained).")
    parser.add_argument("--ablation_epochs_cls", type=int, default=5)
    parser.add_argument("--ablation_epochs_sr", type=int, default=5)
    # Mask-ratio ablations
    parser.add_argument("--run_mask_ablation", action="store_true",
                        help="Run MAE mask-ratio experiments.")
    parser.add_argument("--mask_ratios", type=str, default="0.5,0.75,0.9",
                        help="Comma-separated list of MAE mask ratios for ablation.")
    parser.add_argument("--mask_ablation_mae_epochs", type=int, default=5)
    parser.add_argument("--mask_ablation_cls_epochs", type=int, default=5)
    parser.add_argument("--mask_ablation_sr_epochs", type=int, default=5)

    # Notebook-safe: ignore unknown IPython args
    if hasattr(sys, "argv"):
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"[WARN] Ignoring unknown arguments: {unknown}")
    else:
        args = parser.parse_args([])

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # All paths relative to data_root
    os.chdir(args.data_root)
    os.makedirs("outputs", exist_ok=True)

    # Extract datasets if ZIP exists
    extract_zip_if_exists("Dataset1.zip", ".")
    extract_zip_if_exists("Dataset2.zip", ".")

    dataset1_root = find_dataset_root_for_classes(".", ["axion", "cdm", "no_sub"])
    if dataset1_root is None:
        raise RuntimeError("Could not locate Dataset1 root containing axion/cdm/no_sub directories.")

    dataset2_root = find_dataset_root_for_classes(".", ["HR", "LR"])
    if dataset2_root is None:
        raise RuntimeError("Could not locate Dataset2 root containing HR/LR directories.")

    mae_ds, train_cls_ds, val_cls_ds, class_to_idx = build_task_VI_A_datasets(
        dataset1_root=dataset1_root,
        val_fraction=0.1,
        target_size=64,
        seed=args.seed,
    )
    train_sr_ds, val_sr_ds = build_task_VI_B_datasets(
        dataset2_root=dataset2_root,
        val_fraction=0.1,
        target_size=64,
        seed=args.seed,
    )

    class_names = [None] * len(class_to_idx)
    for k, v in class_to_idx.items():
        class_names[v] = k

    mae_loader = DataLoader(mae_ds,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True)

    # --------- Phase 1: MAE pre-training ----------
    encoder = ViTEncoder(img_size=64,
                         patch_size=4,
                         in_chans=1,
                         embed_dim=192,
                         depth=6,
                         num_heads=3,
                         mlp_ratio=4.0,
                         drop_rate=0.0)
    mae_model = MaskedAutoencoderViT(encoder=encoder,
                                     img_size=64,
                                     patch_size=4,
                                     in_chans=1,
                                     mask_ratio=args.mae_mask_ratio,
                                     decoder_dim=192,
                                     decoder_depth=2,
                                     decoder_num_heads=3)
    print(f"[INFO] Starting MAE pre-training on no_sub class (mask_ratio={args.mae_mask_ratio})...")
    _ = train_mae(mae_model,
                  mae_loader,
                  device=device,
                  epochs=args.mae_epochs,
                  lr=args.mae_lr)

    mae_encoder_state = mae_model.encoder.state_dict()
    torch.save(mae_encoder_state, os.path.join("outputs", "mae_encoder.pth"))
    print("[INFO] MAE encoder weights saved to outputs/mae_encoder.pth")

    # --------- Phase 2: Classification ----------
    print("[INFO] Starting classification fine-tuning...")
    if args.use_optuna:
        study_cls = optuna.create_study(direction="maximize")
        study_cls.optimize(
            lambda trial: objective_classification(
                trial,
                train_cls_ds,
                val_cls_ds,
                mae_encoder_state,
                num_classes=len(class_names),
                device=device,
                batch_size=args.batch_size,
            ),
            n_trials=args.optuna_trials_cls,
        )
        best_cls_params = study_cls.best_params
        print(f"[Optuna CLS] Best params: {best_cls_params}")
        cls_lr = best_cls_params["lr"]
        cls_wd = best_cls_params["weight_decay"]
        cls_drop = best_cls_params["drop_rate"]
    else:
        cls_lr = 5e-5
        cls_wd = 1e-5
        cls_drop = 0.1

    encoder_cls = ViTEncoder(img_size=64,
                             patch_size=4,
                             in_chans=1,
                             embed_dim=192,
                             depth=6,
                             num_heads=3,
                             mlp_ratio=4.0,
                             drop_rate=cls_drop)
    encoder_cls.load_state_dict(mae_encoder_state)
    classifier = ViTClassifier(encoder_cls,
                               num_classes=len(class_names)).to(device)

    train_loader_cls = DataLoader(train_cls_ds,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)
    val_loader_cls = DataLoader(val_cls_ds,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True)

    optimizer_cls = torch.optim.Adam(classifier.parameters(),
                                     lr=cls_lr,
                                     weight_decay=cls_wd)
    criterion_cls = nn.CrossEntropyLoss()
    num_epochs_cls = 10

    for epoch in range(num_epochs_cls):
        classifier.train()
        running_loss = 0.0
        pbar = tqdm(train_loader_cls,
                    desc=f"CLS Epoch {epoch+1}/{num_epochs_cls}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer_cls.zero_grad()
            logits, _ = classifier(imgs)
            loss = criterion_cls(logits, labels)
            loss.backward()
            optimizer_cls.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})
        epoch_loss = running_loss / len(train_loader_cls.dataset)
        print(f"[CLS] Epoch {epoch+1}/{num_epochs_cls} Loss: {epoch_loss:.6f}")

    cls_metrics = evaluate_classifier(classifier,
                                      val_loader_cls,
                                      device,
                                      num_classes=len(class_names))
    print(f"[CLS] Validation Macro AUC: {cls_metrics['auc_macro']:.4f}, "
          f"ACC: {cls_metrics['accuracy']:.4f}, F1: {cls_metrics['f1_macro']:.4f}")
    plot_roc_curves(cls_metrics,
                    class_names=class_names,
                    save_path=os.path.join("outputs", "roc_curve.png"))
    plot_tsne(cls_metrics["features"],
              cls_metrics["labels"],
              class_names=class_names,
              save_path=os.path.join("outputs", "tsne.png"))
    plot_confusion_and_report(cls_metrics,
                              class_names=class_names,
                              prefix=os.path.join("outputs", "cls_pretrained"))
    plot_reliability_diagram(cls_metrics,
                             save_path=os.path.join("outputs", "reliability_pretrained.png"))

    torch.save(classifier.state_dict(),
               os.path.join("outputs", "classifier.pth"))
    print("[INFO] Classifier weights saved to outputs/classifier.pth")

    # --------- Phase 3: Super-resolution ----------
    print("[INFO] Starting super-resolution fine-tuning...")
    if args.use_optuna:
        study_sr = optuna.create_study(direction="maximize")
        study_sr.optimize(
            lambda trial: objective_sr(
                trial,
                train_sr_ds,
                val_sr_ds,
                mae_encoder_state,
                device=device,
                batch_size=args.batch_size,
            ),
            n_trials=args.optuna_trials_sr,
        )
        best_sr_params = study_sr.best_params
        print(f"[Optuna SR] Best params: {best_sr_params}")
        sr_lr = best_sr_params["lr"]
        sr_wd = best_sr_params["weight_decay"]
        sr_drop = best_sr_params["drop_rate"]
    else:
        sr_lr = 5e-5
        sr_wd = 1e-5
        sr_drop = 0.1

    encoder_sr = ViTEncoder(img_size=64,
                            patch_size=4,
                            in_chans=1,
                            embed_dim=192,
                            depth=6,
                            num_heads=3,
                            mlp_ratio=4.0,
                            drop_rate=sr_drop)
    encoder_sr.load_state_dict(mae_encoder_state)
    sr_model = ViTSuperResolution(encoder_sr).to(device)

    train_loader_sr = DataLoader(train_sr_ds,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 pin_memory=True)
    val_loader_sr = DataLoader(val_sr_ds,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=2,
                               pin_memory=True)

    optimizer_sr = torch.optim.Adam(sr_model.parameters(),
                                    lr=sr_lr,
                                    weight_decay=sr_wd)
    criterion_sr = nn.MSELoss()
    num_epochs_sr = 10

    for epoch in range(num_epochs_sr):
        sr_model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader_sr,
                    desc=f"SR Epoch {epoch+1}/{num_epochs_sr}")
        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            optimizer_sr.zero_grad()
            pred = sr_model(lr_img)
            loss = criterion_sr(pred, hr_img)
            loss.backward()
            optimizer_sr.step()
            running_loss += loss.item() * lr_img.size(0)
            pbar.set_postfix({"loss": loss.item()})
        epoch_loss = running_loss / len(train_loader_sr.dataset)
        print(f"[SR] Epoch {epoch+1}/{num_epochs_sr} Loss: {epoch_loss:.6f}")

    sr_metrics = evaluate_sr(sr_model, val_loader_sr, device)
    visualize_sr_example(sr_model,
                         val_loader_sr,
                         device,
                         save_path=os.path.join("outputs", "sr_example.png"))
    visualize_sr_grid(sr_model,
                      val_loader_sr,
                      device,
                      n_examples=6,
                      save_path=os.path.join("outputs", "sr_grid.png"))
    torch.save(sr_model.state_dict(),
               os.path.join("outputs", "sr_model.pth"))
    print("[INFO] SR model weights saved to outputs/sr_model.pth")

    # --------- Ablations ----------
    if args.run_ablation:
        run_ablation_experiments(mae_encoder_state,
                                 train_cls_ds,
                                 val_cls_ds,
                                 train_sr_ds,
                                 val_sr_ds,
                                 class_names,
                                 device,
                                 batch_size=args.batch_size,
                                 epochs_cls=args.ablation_epochs_cls,
                                 epochs_sr=args.ablation_epochs_sr,
                                 cls_lr=cls_lr,
                                 cls_wd=cls_wd,
                                 cls_drop=cls_drop,
                                 sr_lr=sr_lr,
                                 sr_wd=sr_wd,
                                 sr_drop=sr_drop)

    # --------- MAE mask-ratio ablations ----------
    if args.run_mask_ablation:
        ratios = [float(r.strip()) for r in args.mask_ratios.split(",") if r.strip()]
        run_mask_ratio_experiments(
            mask_ratios=ratios,
            mae_loader=mae_loader,
            train_cls_ds=train_cls_ds,
            val_cls_ds=val_cls_ds,
            train_sr_ds=train_sr_ds,
            val_sr_ds=val_sr_ds,
            class_names=class_names,
            device=device,
            batch_size=args.batch_size,
            mae_epochs=args.mask_ablation_mae_epochs,
            mae_lr=args.mae_lr,
            cls_epochs=args.mask_ablation_cls_epochs,
            cls_lr=cls_lr,
            cls_wd=cls_wd,
            cls_drop=cls_drop,
            sr_epochs=args.mask_ablation_sr_epochs,
            sr_lr=sr_lr,
            sr_wd=sr_wd,
            sr_drop=sr_drop,
        )


if __name__ == "__main__":
    main()
