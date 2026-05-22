"""
src/training/loss.py
====================
Funciones de pérdida para segmentación semántica con desequilibrio de clases.

Clases
------
DiceLoss     -- pérdida basada en el coeficiente Dice por clase
CombinedLoss -- combinación ponderada de Dice + Cross-Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Pérdida Dice para segmentación semántica multi-clase.

    El coeficiente Dice mide el solapamiento entre la predicción y el
    ground truth. Es especialmente útil en imágenes médicas donde las
    clases de interés (tumor) ocupan una fracción muy pequeña del volumen
    total, lo que hace que la Cross-Entropy sola favorezca al fondo.

    Fórmula por clase c:
        Dice_c = (2 · Σ(p_c · g_c) + ε) / (Σp_c + Σg_c + ε)
    Pérdida: 1 - mean(Dice_c)

    Parameters
    ----------
    num_classes : número de clases
    smooth      : término de suavizado ε para evitar división por cero
    ignore_bg   : si True, excluye la clase 0 (fondo) del cálculo
    """

    def __init__(self, num_classes: int = 4, smooth: float = 1e-6, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ignore_bg   = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W) — salida del modelo sin softmax
        targets : (B, H, W)    — etiquetas enteras

        Returns
        -------
        torch.Tensor : escalar con la pérdida Dice
        """
        probs      = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        start = 1 if self.ignore_bg else 0
        dice_vals = []
        for c in range(start, self.num_classes):
            p    = probs[:, c]
            g    = targets_oh[:, c]
            dice = (2 * (p * g).sum() + self.smooth) / (p.sum() + g.sum() + self.smooth)
            dice_vals.append(dice)

        return 1.0 - torch.stack(dice_vals).mean()


class CombinedLoss(nn.Module):
    """
    Pérdida combinada: α·Dice + β·CrossEntropy.

    La Cross-Entropy proporciona gradientes estables en las primeras
    épocas, mientras que Dice es invariante al desequilibrio de clases y
    optimiza directamente la métrica de evaluación.

    Usar ambas en combinación mejora la convergencia y el rendimiento
    final en benchmarks de segmentación médica.

    Parameters
    ----------
    num_classes : número de clases
    dice_w      : peso de la pérdida Dice (α)
    ce_w        : peso de la Cross-Entropy (β)
    """

    def __init__(self, num_classes: int = 4, dice_w: float = 0.5, ce_w: float = 0.5):
        super().__init__()
        self.dice   = DiceLoss(num_classes)
        self.ce     = nn.CrossEntropyLoss()
        self.dice_w = dice_w
        self.ce_w   = ce_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self.dice(logits, targets) + self.ce_w * self.ce(logits, targets)
