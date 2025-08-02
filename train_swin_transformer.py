import json
import logging
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================================================================================
# PHASE I: ENHANCED CONFIGURATION FOR v2
# ================================================================================


@dataclass
class ModelConfigV2:
    """
    Enhanced configuration for v2 Pneumonia Detection System.

    Strategic Response to v1 Weaknesses:
    - Swin Transformer architecture for hierarchical attention
    - Focal Loss parameters to combat class imbalance
    - CutMix integration for overfitting prevention
    - Cross-validation for robust model selection
    """

    # Model Architecture - STRATEGIC PIVOT TO SWIN
    model_name: str = "swin_base_patch4_window7_224"  # Hierarchical attention advantage
    pretrained: bool = True
    num_classes: int = 2
    image_size: int = 224

    # Training Parameters - ENHANCED FOR v2
    batch_size: int = 32  # Optimized for M1 Max memory
    num_epochs: int = 10  # Reduced from v1 based on overfitting observations
    learning_rate: float = 2e-4  # Slightly more conservative for Swin
    weight_decay: float = 0.02  # Increased regularization
    warmup_epochs: int = 5

    # FOCAL LOSS CONFIGURATION - Class Imbalance Combat
    focal_alpha: float = (
        0.6  # Weight for majority class (pneumonia) - Optimized for 74/26 split
    )
    focal_gamma: float = (
        1.2  # Focus on hard examples - Reduced to prevent overconfidence
    )

    # CUTMIX CONFIGURATION - Overfitting Prevention
    cutmix_prob: float = 0.5  # Probability of applying CutMix
    cutmix_alpha: float = 1.0  # Beta distribution parameter

    # CROSS-VALIDATION STRATEGY
    n_folds: int = 5  # 5-fold CV for robust validation
    cv_random_seed: int = 42  # Reproducible splits

    # Data Pipeline
    num_workers: int = 8  # M1 Max optimization

    # Hardware Optimization
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    mixed_precision: bool = False  # Enable when MPS AMP stabilizes

    # Paths
    data_path: str = "./data"
    output_dir: str = "./outputs_v2"
    checkpoint_dir: str = "./checkpoints_v2"
    logs_dir: str = "./logs_v2"

    # Clinical Validation
    grad_cam_samples: int = 8
    confidence_threshold: float = 0.5

    def __post_init__(self):
        """Create necessary directories for v2."""
        for dir_path in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config_v2 = ModelConfigV2()

# ================================================================================
# PHASE I: ENHANCED DATASET WITH CUTMIX SUPPORT
# ================================================================================


class ChestXrayDatasetV2(Dataset):
    """
    Enhanced Dataset for v2 with CutMix support and improved medical preprocessing.

    Key v2 Enhancements:
    - CutMix augmentation integration
    - Better class distribution logging
    - Optimized for cross-validation splits
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Any] = None,
        target_size: Tuple[int, int] = (224, 224),
        cutmix_prob: float = 0.0,
        cutmix_alpha: float = 1.0,
    ):
        """
        Initialize enhanced dataset.

        Args:
            data_dir: Root directory containing train/val/test folders
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            target_size: Target image dimensions
            cutmix_prob: Probability of applying CutMix (train only)
            cutmix_alpha: Beta distribution parameter for CutMix
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.cutmix_prob = cutmix_prob if split == "train" else 0.0
        self.cutmix_alpha = cutmix_alpha

        # Build file paths and labels
        self.samples = self._build_samples()
        self.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        self._log_class_distribution()

    def _build_samples(self) -> List[Tuple[Path, str]]:
        """Build list of (image_path, label) tuples."""
        samples = []
        split_dir = self.data_dir / self.split

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue

            class_name = class_dir.name
            for img_path in class_dir.glob("*.jpeg"):
                samples.append((img_path, class_name))

        return samples

    def _log_class_distribution(self):
        """Enhanced class distribution logging for v2."""
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        total_samples = len(self.samples)
        logger.info(f"=== CLASS DISTRIBUTION - {self.split.upper()} ===")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"  {class_name}: {count:4d} samples ({percentage:5.1f}%)")

        # Calculate imbalance ratio for Focal Loss tuning
        if len(class_counts) == 2:
            normal_count = class_counts.get("NORMAL", 0)
            pneumonia_count = class_counts.get("PNEUMONIA", 0)
            if normal_count > 0 and pneumonia_count > 0:
                imbalance_ratio = max(normal_count, pneumonia_count) / min(
                    normal_count, pneumonia_count
                )
                logger.info(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_cutmix(
        self, image1: torch.Tensor, label1: int, image2: torch.Tensor, label2: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix augmentation.

        CutMix Strategy: Combines patches from two images to create robust features
        and prevent overfitting - direct response to v1 overfitting observations.
        """
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1

        batch_size = image1.size(0) if image1.dim() == 4 else 1
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)

        H, W = image1.size(2), image1.size(3)

        # Generate random box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        mixed_image = image1.clone()
        mixed_image[:, :, bby1:bby2, bbx1:bbx2] = image2[:, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # Create mixed label
        mixed_label = torch.zeros(2)
        mixed_label[label1] = lam
        mixed_label[label2] = 1 - lam

        return mixed_image.squeeze(0), mixed_label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced getitem with CutMix support."""
        img_path, label = self.samples[idx]

        # Load and preprocess image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Fallback transforms
            image = cv2.resize(image, self.target_size)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_idx = self.class_to_idx[label]

        # Apply CutMix during training
        if self.cutmix_prob > 0 and random.random() < self.cutmix_prob:
            # Get another random sample for mixing
            mix_idx = random.randint(0, len(self.samples) - 1)
            mix_img_path, mix_label = self.samples[mix_idx]

            # Load second image
            mix_image = cv2.imread(str(mix_img_path))
            mix_image = cv2.cvtColor(mix_image, cv2.COLOR_BGR2RGB)

            if self.transform:
                mix_transformed = self.transform(image=mix_image)
                mix_image = mix_transformed["image"]
            else:
                mix_image = cv2.resize(mix_image, self.target_size)
                mix_image = torch.from_numpy(mix_image).permute(2, 0, 1).float() / 255.0

            mix_label_idx = self.class_to_idx[mix_label]

            # Apply CutMix
            mixed_image, mixed_label = self._apply_cutmix(
                image, label_idx, mix_image, mix_label_idx
            )
            return mixed_image, mixed_label
        else:
            # Standard one-hot encoding
            one_hot_label = torch.zeros(2)
            one_hot_label[label_idx] = 1.0
            return image, one_hot_label

    def get_sample_for_visualization(self, idx: int) -> Tuple[np.ndarray, str, Path]:
        """Get raw sample for visualization purposes."""
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label, img_path


class MedicalImageTransformsV2:
    """Enhanced medical imaging transforms for v2."""

    @staticmethod
    def get_train_transforms(image_size: int = 224) -> A.Compose:
        """
        Enhanced training augmentations for v2.
        More aggressive augmentations to combat overfitting observed in v1.
        """
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                # Enhanced geometric augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1,  # Increased from v1
                    scale_limit=0.15,  # Increased from v1
                    rotate_limit=15,  # Increased from v1
                    p=0.4,  # Increased probability
                ),
                A.HorizontalFlip(p=0.5),
                # Enhanced intensity augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,  # Increased from v1
                    contrast_limit=0.2,  # Increased from v1
                    p=0.5,  # Increased probability
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=0.4),  # Wider range
                # Additional augmentations for robustness
                A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),  # Increased
                A.MotionBlur(blur_limit=5, p=0.15),  # Increased
                A.ElasticTransform(alpha=50, sigma=5, p=0.1),  # New for v2
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def get_val_transforms(image_size: int = 224) -> A.Compose:
        """Validation/test transforms (no augmentation)."""
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


# ================================================================================
# PHASE II: FOCAL LOSS IMPLEMENTATION - CLASS IMBALANCE COMBAT
# ================================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.

    STRATEGIC JUSTIFICATION:
    - Direct response to 74% vs 26% class imbalance identified in v1
    - Focuses training on hard-to-classify examples
    - Reduces contribution of easy examples that dominate standard CE loss
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (should favor NORMAL class)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        logger.info(f"FocalLoss initialized: alpha={alpha}, gamma={gamma}")
        logger.info("Strategic Response: Combat 74% PNEUMONIA vs 26% NORMAL imbalance")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.

        Args:
            inputs: Model predictions [batch_size, num_classes] or [batch_size, 2]
            targets: Ground truth labels [batch_size, num_classes] (one-hot or soft labels)
        """
        # Handle both hard labels and soft labels (from CutMix)
        if targets.dim() == 1:
            # Convert hard labels to one-hot
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        else:
            targets_one_hot = targets

        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets_one_hot, reduction="none")

        # Calculate p_t
        pt = torch.exp(-ce_loss)

        # Calculate alpha_t (class weighting)
        if targets.dim() == 1:
            alpha_t = (
                self.alpha * targets_one_hot[:, 0]
                + (1 - self.alpha) * targets_one_hot[:, 1]
            )
        else:
            alpha_t = (
                self.alpha * targets_one_hot[:, 0]
                + (1 - self.alpha) * targets_one_hot[:, 1]
            )

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ================================================================================
# PHASE II: SWIN TRANSFORMER MODEL ARCHITECTURE
# ================================================================================


class PneumoniaSwinV2(nn.Module):
    """
    Swin Transformer for Pneumonia Classification - v2 Architecture.

    STRATEGIC ADVANTAGE OVER v1 ViT:
    - Hierarchical attention: Better multi-scale feature extraction
    - Window-based attention: More efficient than global attention
    - Shifted windows: Better feature locality for medical imaging
    """

    def __init__(self, config: ModelConfigV2):
        super().__init__()
        self.config = config

        # Load pre-trained Swin Transformer
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",  # Global average pooling
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Enhanced classification head for medical imaging
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.2),  # Increased dropout vs v1
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 2, config.num_classes),
        )

        # Initialize weights
        self._init_weights()

        logger.info(f"Swin Transformer v2 initialized: {config.model_name}")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        """Initialize classification head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features from Swin backbone
        features = self.backbone(x)  # [B, feature_dim]

        # Classification
        logits = self.classifier(features)
        return logits


# ================================================================================
# PHASE II: ENHANCED TRAINING ENGINE WITH CROSS-VALIDATION
# ================================================================================


class PneumoniaTrainerV2:
    """
    Enhanced training engine with cross-validation and advanced techniques.

    KEY v2 ENHANCEMENTS:
    - 5-fold cross-validation for robust model selection
    - Focal Loss integration
    - CutMix-aware training loop
    - Enhanced monitoring and logging
    """

    def __init__(self, config: ModelConfigV2):
        self.config = config
        self.device = torch.device(config.device)

        logger.info(f"Enhanced Trainer v2 initialized on device: {self.device}")
        logger.info("Key Enhancements: Cross-validation, Focal Loss, CutMix support")

    def _create_model(self) -> PneumoniaSwinV2:
        """Create and initialize model."""
        return PneumoniaSwinV2(self.config).to(self.device)

    def _create_criterion(self) -> FocalLoss:
        """Create Focal Loss criterion."""
        return FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)

    def _create_optimizer(self, model: PneumoniaSwinV2) -> torch.optim.Optimizer:
        """Create optimizer."""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.config.num_epochs // 4, T_mult=2
        )

    def train_single_fold(
        self, train_loader: DataLoader, val_loader: DataLoader, fold: int
    ) -> Dict[str, Any]:
        """
        Train a single fold of cross-validation.

        Returns:
            Dictionary containing training metrics and best model state
        """
        logger.info(f"=== FOLD {fold + 1}/{self.config.n_folds} ===")

        # Initialize model, criterion, optimizer, scheduler
        model = self._create_model()
        criterion = self._create_criterion()
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)

        # TensorBoard logging for this fold
        writer = SummaryWriter(Path(self.config.logs_dir) / f"fold_{fold}")

        # Metrics tracking with early stopping
        best_val_acc = 0.0
        best_model_state = None
        best_epoch = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Early stopping parameters (consistent with final training)
        patience = 3  # Slightly lower for CV folds
        min_delta = 0.01  # Minimum improvement threshold
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer
            )

            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)

            # Learning rate scheduling
            scheduler.step()

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            # TensorBoard logging
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

            # Save best model for this fold with early stopping logic
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
                logger.debug(f"  New best model at epoch {epoch+1}: {val_acc:.2f}%")
            else:
                patience_counter += 1
                logger.debug(
                    f"  No improvement. Patience: {patience_counter}/{patience}"
                )

            # Early stopping check
            early_stop = False
            if patience_counter >= patience:
                logger.info(
                    f"  Early stopping triggered at epoch {epoch+1} (patience={patience})"
                )
                early_stop = True

            # Safety check for erratic validation performance
            elif epoch > 3 and val_acc < 30.0:  # Very low threshold for CV
                logger.warning(
                    f"  Validation performance severely degraded. Early stopping at epoch {epoch+1}"
                )
                early_stop = True

            # Logging (every epoch)
            logger.info(
                f"Fold {fold+1}, Epoch {epoch+1:2d}/{self.config.num_epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
                f"Best: {best_val_acc:.1f}% (Epoch {best_epoch+1}) | "
                f"Patience: {patience_counter}/{patience}"
            )

            # Break if early stopping triggered
            if early_stop:
                break

        writer.close()

        # Log final results for this fold
        logger.info(f"Fold {fold+1} completed after {best_epoch+1} epochs:")
        logger.info(f"  Best Val Acc: {best_val_acc:.2f}%")
        logger.info(
            f"  Early stopped: {'Yes' if best_epoch < self.config.num_epochs - 1 else 'No'}"
        )

        return {
            "fold": fold,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,  # Important: Include best epoch for final model CV guidance
            "best_model_state": best_model_state,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "early_stopped": best_epoch < self.config.num_epochs - 1,
            "total_epochs": best_epoch + 1,
        }

    def _train_epoch(
        self,
        model: PneumoniaSwinV2,
        train_loader: DataLoader,
        criterion: FocalLoss,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """Train for one epoch with CutMix support."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Handle CutMix labels (soft labels) vs standard labels
            if labels.dim() > 1 and labels.size(1) > 1:
                # Soft labels from CutMix
                loss = criterion(outputs, labels)
                # For accuracy calculation, use the class with higher probability
                _, target_class = torch.max(labels, 1)
            else:
                # Standard hard labels
                if labels.dim() > 1:
                    labels = torch.argmax(labels, dim=1)
                target_class = labels
                # Convert to one-hot for focal loss
                labels_one_hot = F.one_hot(labels, num_classes=2).float()
                loss = criterion(outputs, labels_one_hot)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_class.size(0)
            correct += (predicted == target_class).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100 * correct / total:.2f}%"}
            )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _validate_epoch(
        self, model: PneumoniaSwinV2, val_loader: DataLoader, criterion: FocalLoss
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)

                # Handle labels (should be one-hot from dataset)
                if labels.dim() > 1 and labels.size(1) > 1:
                    loss = criterion(outputs, labels)
                    _, target_class = torch.max(labels, 1)
                else:
                    if labels.dim() > 1:
                        labels = torch.argmax(labels, dim=1)
                    target_class = labels
                    labels_one_hot = F.one_hot(labels, num_classes=2).float()
                    loss = criterion(outputs, labels_one_hot)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target_class.size(0)
                correct += (predicted == target_class).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def run_cross_validation(self, train_dataset: ChestXrayDatasetV2) -> Dict[str, Any]:
        """
        Run 5-fold cross-validation on training data.

        STRATEGIC PURPOSE:
        - Robust model selection despite 16-sample val set instability
        - Better understanding of model performance variance
        - Guide hyperparameter optimization in future sprints
        """
        logger.info("=== STARTING 5-FOLD CROSS-VALIDATION ===")
        logger.info("Strategic Purpose: Overcome 16-sample validation instability")

        # Create 5-fold splits
        kfold = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.cv_random_seed,
        )
        indices = list(range(len(train_dataset)))

        cv_results = []
        fold_performances = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            # Create fold datasets
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)

            # Create data loaders for this fold
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

            # Train this fold
            fold_result = self.train_single_fold(train_loader, val_loader, fold)
            cv_results.append(fold_result)
            fold_performances.append(fold_result["best_val_acc"])

            logger.info(
                f"Fold {fold+1} completed: Best Val Acc = {fold_result['best_val_acc']:.2f}%"
            )

        # Calculate cross-validation statistics
        mean_performance = np.mean(fold_performances)
        std_performance = np.std(fold_performances)

        # Extract early stopping statistics
        early_stopped_folds = sum(
            1 for result in cv_results if result.get("early_stopped", False)
        )
        fold_epochs = [
            result.get("total_epochs", self.config.num_epochs) for result in cv_results
        ]
        mean_epochs = np.mean(fold_epochs)

        logger.info("=== CROSS-VALIDATION RESULTS ===")
        logger.info(
            f"Mean CV Performance: {mean_performance:.2f} ± {std_performance:.2f}%"
        )
        logger.info("Early Stopping Statistics:")
        logger.info(
            f"  Folds with early stopping: {early_stopped_folds}/{self.config.n_folds}"
        )
        logger.info(f"  Average epochs per fold: {mean_epochs:.1f}")
        logger.info("Individual Fold Performance:")
        for i, (perf, result) in enumerate(zip(fold_performances, cv_results)):
            epochs = result.get("total_epochs", self.config.num_epochs)
            early_stop_status = "ES" if result.get("early_stopped", False) else "Full"
            logger.info(
                f"  Fold {i+1}: {perf:.2f}% ({epochs} epochs, {early_stop_status})"
            )

        return {
            "cv_results": cv_results,
            "mean_performance": mean_performance,
            "std_performance": std_performance,
            "fold_performances": fold_performances,
        }

    def train_final_model(
        self,
        full_train_loader: DataLoader,
        official_val_loader: DataLoader,
        cv_results: Optional[Dict[str, Any]] = None,
    ) -> PneumoniaSwinV2:
        """
        Train final model with CV-guided early stopping and robust epoch selection.

        STRATEGIC PURPOSE:
        - Use all training data for final model
        - CV-guided early stopping to overcome 16-sample val instability
        - Robust decision logic for final model selection
        """
        logger.info("=== TRAINING FINAL MODEL WITH CV GUIDANCE ===")
        logger.info("Strategy: CV-guided early stopping + robust model selection")

        # Initialize final model
        model = self._create_model()
        criterion = self._create_criterion()
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)

        # TensorBoard logging
        writer = SummaryWriter(Path(self.config.logs_dir) / "final_model")

        # Enhanced tracking with CV guidance
        best_val_acc = 0.0
        best_model_state = None
        best_epoch = 0

        # CV-guided parameters
        patience = 5
        min_delta = 0.02
        patience_counter = 0
        cv_guided_epoch = 7  # Default fallback
        cv_guided_state = None

        # Extract CV insights if available
        if cv_results and "cv_results" in cv_results:
            fold_best_epochs = []
            early_stopped_count = 0

            for fold_result in cv_results["cv_results"]:
                if "best_epoch" in fold_result:
                    fold_best_epochs.append(fold_result["best_epoch"])
                if fold_result.get("early_stopped", False):
                    early_stopped_count += 1

            if fold_best_epochs:
                cv_guided_epoch = int(np.mean(fold_best_epochs))
                cv_epoch_std = np.std(fold_best_epochs)
                logger.info("CV insights:")
                logger.info(f"  Optimal epoch: {cv_guided_epoch} ± {cv_epoch_std:.1f}")
                logger.info(
                    f"  Early stopped folds: {early_stopped_count}/{len(cv_results['cv_results'])}"
                )

                # Adjust patience based on CV early stopping frequency
                if early_stopped_count >= 3:  # Most folds early stopped
                    patience = 4  # More conservative
                    logger.info(
                        f"  Reduced patience to {patience} (many CV folds early stopped)"
                    )
                else:
                    logger.info(f"  Using standard patience: {patience}")
            else:
                logger.warning("No CV epoch information available. Using defaults.")

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, full_train_loader, criterion, optimizer
            )

            # Validation phase
            val_loss, val_acc = self._validate_epoch(
                model, official_val_loader, criterion
            )

            # Learning rate scheduling
            scheduler.step()

            # TensorBoard logging
            writer.add_scalar("Final/Loss_Train", train_loss, epoch)
            writer.add_scalar("Final/Loss_Validation", val_loss, epoch)
            writer.add_scalar("Final/Accuracy_Train", train_acc, epoch)
            writer.add_scalar("Final/Accuracy_Validation", val_acc, epoch)
            writer.add_scalar(
                "Final/Learning_Rate", optimizer.param_groups[0]["lr"], epoch
            )

            # Save CV-guided checkpoint
            if epoch == cv_guided_epoch:
                cv_guided_state = deepcopy(model.state_dict())
                logger.info(f"Saved CV-guided model at epoch {epoch+1}")

            # Robust early stopping logic
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping with patience
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

            # Safety check for erratic validation (16-sample instability)
            if epoch > 5 and val_acc < 50.0:
                logger.warning(
                    "Validation performance degraded. Using CV-guided epoch."
                )
                break

            # Logging
            logger.info(
                f"Final Model Epoch {epoch+1:2d}/{self.config.num_epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
                f"Best: {best_val_acc:.1f}% (Epoch {best_epoch+1}) | "
                f"Patience: {patience_counter}/{patience}"
            )

        # Intelligent model selection
        if best_val_acc == 100.0 or best_epoch <= 3 or cv_guided_state is not None:
            logger.warning("Validation appears overfitted. Using CV-guided model.")
            if cv_guided_state is not None:
                model.load_state_dict(cv_guided_state)
                final_epoch = cv_guided_epoch
            else:
                final_epoch = best_epoch
                model.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
            final_epoch = best_epoch

        # Save final model
        final_model_path = (
            Path(self.config.checkpoint_dir)
            / f"final_model_v2_{self.config.focal_alpha}_{self.config.focal_gamma}.pth"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": final_epoch,
                "val_acc": best_val_acc,
                "config": self.config,
                "cv_guided": cv_guided_state is not None,
            },
            final_model_path,
        )

        writer.close()

        logger.info("Final model training completed!")
        logger.info(
            f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {final_epoch+1})"
        )
        logger.info(
            f"Model selection strategy: {'CV-guided' if cv_guided_state is not None else 'Validation-guided'}"
        )
        logger.info(f"Model saved to: {final_model_path}")

        return model


# ================================================================================
# PHASE III: ENHANCED EVALUATION SYSTEM
# ================================================================================


class PneumoniaEvaluatorV2:
    """Enhanced evaluation suite for v2 with improved metrics and analysis."""

    def __init__(self, model: PneumoniaSwinV2, config: ModelConfigV2):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.class_names = ["Normal", "Pneumonia"]

        logger.info("Enhanced Evaluator v2 initialized")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation with enhanced metrics."""
        logger.info("Starting comprehensive evaluation...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)

                # Handle labels (convert from one-hot if needed)
                if labels.dim() > 1 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)

                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)

        # Calculate comprehensive metrics
        metrics = self._calculate_enhanced_metrics(y_true, y_pred, y_prob)

        # Generate visualizations
        self._plot_enhanced_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_prob[:, 1])
        self._plot_precision_recall_curve(y_true, y_prob[:, 1])

        return metrics

    def _calculate_enhanced_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate enhanced evaluation metrics."""

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "auc_roc": roc_auc_score(y_true, y_prob[:, 1]),
        }

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        for i, class_name in enumerate(self.class_names):
            metrics[f"precision_{class_name.lower()}"] = precision_per_class[i]
            metrics[f"recall_{class_name.lower()}"] = recall_per_class[i]
            metrics[f"f1_{class_name.lower()}"] = f1_per_class[i]

        # Class-specific analysis for medical context
        normal_recall = recall_per_class[0]  # Sensitivity for Normal
        pneumonia_recall = recall_per_class[1]  # Sensitivity for Pneumonia
        normal_precision = precision_per_class[0]  # PPV for Normal
        pneumonia_precision = precision_per_class[1]  # PPV for Pneumonia

        # Medical interpretation
        metrics["sensitivity_normal"] = normal_recall
        metrics["sensitivity_pneumonia"] = pneumonia_recall
        metrics["ppv_normal"] = normal_precision
        metrics["ppv_pneumonia"] = pneumonia_precision

        # Specificity calculations
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity_normal"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["specificity_pneumonia"] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Print enhanced report
        logger.info("=== ENHANCED EVALUATION RESULTS v2 ===")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")

        logger.info("\n=== CLINICAL METRICS ===")
        logger.info("Normal Detection:")
        logger.info(f"  Sensitivity (Recall): {metrics['sensitivity_normal']:.4f}")
        logger.info(f"  Specificity: {metrics['specificity_normal']:.4f}")
        logger.info(f"  PPV (Precision): {metrics['ppv_normal']:.4f}")

        logger.info("Pneumonia Detection:")
        logger.info(f"  Sensitivity (Recall): {metrics['sensitivity_pneumonia']:.4f}")
        logger.info(f"  Specificity: {metrics['specificity_pneumonia']:.4f}")
        logger.info(f"  PPV (Precision): {metrics['ppv_pneumonia']:.4f}")

        return metrics

    def _plot_enhanced_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot enhanced confusion matrix with clinical interpretation."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )

        plt.title(
            "Enhanced Confusion Matrix - v2 Model\nSwin Transformer with Focal Loss",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)

        # Add clinical interpretation
        tn, fp, fn, tp = cm.ravel()

        # Add text annotations with clinical meaning
        plt.text(
            0.5,
            -0.15,
            f"TN: {tn} (Correct Normal)",
            ha="center",
            transform=plt.gca().transAxes,
        )
        plt.text(
            1.5,
            -0.15,
            f"FP: {fp} (False Pneumonia)",
            ha="center",
            transform=plt.gca().transAxes,
        )
        plt.text(
            0.5,
            -0.25,
            f"FN: {fn} (Missed Pneumonia)",
            ha="center",
            transform=plt.gca().transAxes,
        )
        plt.text(
            1.5,
            -0.25,
            f"TP: {tp} (Correct Pneumonia)",
            ha="center",
            transform=plt.gca().transAxes,
        )

        plt.tight_layout()
        plt.savefig(
            Path(self.config.output_dir) / "enhanced_confusion_matrix_v2.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Plot ROC curve with v2 branding."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            linewidth=3,
            label=f"Swin Transformer v2 (AUC = {auc:.3f})",
            color="darkblue",
        )
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(
            "ROC Curve - Pneumonia Detection v2\nSwin Transformer + Focal Loss + CutMix",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            Path(self.config.output_dir) / "roc_curve_v2.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Plot Precision-Recall curve with v2 enhancements."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=3, color="darkgreen")
        plt.fill_between(recall, precision, alpha=0.3, color="darkgreen")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall (Sensitivity)", fontsize=12)
        plt.ylabel("Precision (PPV)", fontsize=12)
        plt.title(
            "Precision-Recall Curve - Pneumonia Detection v2\nOptimized for Clinical Performance",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            Path(self.config.output_dir) / "precision_recall_curve_v2.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


# ================================================================================
# PHASE IV: DATA PIPELINE CREATION
# ================================================================================


def create_data_loaders_v2(
    config: ModelConfigV2,
) -> Tuple[DataLoader, DataLoader, DataLoader, ChestXrayDatasetV2]:
    """Create enhanced data loaders for v2."""

    # Initialize transforms
    train_transform = MedicalImageTransformsV2.get_train_transforms(config.image_size)
    val_transform = MedicalImageTransformsV2.get_val_transforms(config.image_size)

    # Create datasets with CutMix support
    train_dataset = ChestXrayDatasetV2(
        config.data_path,
        split="train",
        transform=train_transform,
        cutmix_prob=config.cutmix_prob,
        cutmix_alpha=config.cutmix_alpha,
    )

    val_dataset = ChestXrayDatasetV2(
        config.data_path, split="val", transform=val_transform
    )

    test_dataset = ChestXrayDatasetV2(
        config.data_path, split="test", transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset


# ================================================================================
# PHASE IV: OPTUNA-READY TRIAL FUNCTION
# ================================================================================


def run_trial(config: ModelConfigV2) -> float:
    """
    Optuna-ready trial function for future hyperparameter optimization.

    STRATEGIC PREPARATION:
    - Modular design for easy Optuna integration
    - Returns single metric for optimization
    - Structured for automated hyperparameter search

    Args:
        config: Configuration object with hyperparameters

    Returns:
        Cross-validation mean performance for optimization
    """
    logger.info("Running trial with configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Focal Alpha: {config.focal_alpha}")
    logger.info(f"  Focal Gamma: {config.focal_gamma}")
    logger.info(f"  CutMix Prob: {config.cutmix_prob}")

    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders_v2(
        config
    )

    # Initialize trainer
    trainer = PneumoniaTrainerV2(config)

    # Run cross-validation
    cv_results = trainer.run_cross_validation(train_dataset)

    # Return mean performance for optimization
    return cv_results["mean_performance"]


# ================================================================================
# PHASE IV: MAIN EXECUTION PIPELINE v2
# ================================================================================


def main():
    """
    Main execution pipeline for v2 Pneumonia Detection System.

    STRATEGIC EXECUTION PLAN:
    1. Cross-validation for robust model selection
    2. Final model training on full dataset
    3. Comprehensive evaluation with clinical metrics
    4. Performance comparison with v1 baseline
    """

    print("🏥 SOTA Pneumonia Detection v2 - Swin Transformer")
    print("=" * 70)
    print(f"🖥️  Device: {config_v2.device}")
    print(f"🧠 Model: {config_v2.model_name}")
    print(f"📊 Image Size: {config_v2.image_size}x{config_v2.image_size}")
    print(f"🔄 Epochs: {config_v2.num_epochs}")
    print(f"🎯 Focal Loss: α={config_v2.focal_alpha}, γ={config_v2.focal_gamma}")
    print(f"🔀 CutMix: {config_v2.cutmix_prob * 100:.0f}% probability")
    print(f"📈 Cross-Validation: {config_v2.n_folds} folds")
    print("=" * 70)
    print("🎯 STRATEGIC RESPONSE TO v1 WEAKNESSES:")
    print("   • Swin Transformer: Hierarchical attention for medical features")
    print("   • Focal Loss: Combat 74% vs 26% class imbalance")
    print("   • CutMix: Advanced augmentation against overfitting")
    print("   • 5-Fold CV: Overcome 16-sample validation instability")
    print("=" * 70)

    try:
        # Phase I: Data Pipeline Setup
        logger.info("Phase I: Setting up enhanced data pipeline...")
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders_v2(
            config_v2
        )

        # Phase II: Cross-Validation Training
        logger.info("Phase II: Running cross-validation training...")
        trainer = PneumoniaTrainerV2(config_v2)
        cv_results = trainer.run_cross_validation(train_dataset)

        # Save cross-validation results
        cv_results_path = Path(config_v2.output_dir) / "cross_validation_results.json"
        with open(cv_results_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            cv_results_serializable = deepcopy(cv_results)
            for fold_result in cv_results_serializable["cv_results"]:
                for key in [
                    "train_losses",
                    "val_losses",
                    "train_accuracies",
                    "val_accuracies",
                ]:
                    if key in fold_result:
                        fold_result[key] = [float(x) for x in fold_result[key]]
                # Remove model state (not serializable)
                fold_result.pop("best_model_state", None)

            json.dump(cv_results_serializable, f, indent=2)

        # Phase III: Final Model Training
        logger.info("Phase III: Training final model on full dataset...")
        final_model = trainer.train_final_model(train_loader, val_loader, cv_results)

        # Phase IV: Comprehensive Evaluation
        logger.info("Phase IV: Comprehensive evaluation on test set...")
        evaluator = PneumoniaEvaluatorV2(final_model, config_v2)
        test_metrics = evaluator.evaluate(test_loader)

        # Save evaluation results
        test_metrics_path = (
            Path(config_v2.output_dir) / "test_evaluation_metrics_v2.json"
        )
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

        # Final Results Summary
        logger.info("=" * 70)
        logger.info("🎉 v2 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(
            f"📈 Cross-Validation Performance: {cv_results['mean_performance']:.2f} ± {cv_results['std_performance']:.2f}%"
        )
        logger.info(f"🔍 Test Set Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"🎯 Test Set AUC-ROC: {test_metrics['auc_roc']:.4f}")
        logger.info(
            f"⚕️  Pneumonia Sensitivity: {test_metrics['sensitivity_pneumonia']:.4f}"
        )
        logger.info(f"⚕️  Normal Sensitivity: {test_metrics['sensitivity_normal']:.4f}")
        logger.info("=" * 70)
        logger.info(f"📁 Results saved in: {config_v2.output_dir}")
        logger.info(
            f"💾 Model checkpoint: {config_v2.checkpoint_dir}/final_model_v2.pth"
        )
        logger.info("=" * 70)

        # Performance comparison summary
        logger.info("🏆 KEY v2 IMPROVEMENTS:")
        logger.info("   ✅ Swin Transformer: Better hierarchical feature extraction")
        logger.info("   ✅ Focal Loss: Targeted class imbalance mitigation")
        logger.info("   ✅ CutMix: Advanced augmentation for generalization")
        logger.info("   ✅ Cross-Validation: Robust performance estimation")
        logger.info("   ✅ Enhanced Metrics: Clinical-grade evaluation")

    except Exception as e:
        logger.error(f"❌ v2 Pipeline failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
