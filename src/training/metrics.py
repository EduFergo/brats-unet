"""
src/training/metrics.py
=======================
Métricas de evaluación para segmentación de tumores cerebrales BraTS.

El benchmark BraTS evalúa tres regiones superpuestas que tienen
distinto significado clínico:

  WT (Whole Tumor)     = clases 1 + 2 + 3  → todo el tumor
  TC (Tumor Core)      = clases 1 + 3       → núcleo del tumor
  ET (Enhancing Tumor) = clase 3            → zona de realce (activa)

Esta jerarquía refleja cómo los radiólogos planifican el tratamiento:
WT para delimitar la lesión, TC para la zona de resección quirúrgica,
ET para monitorizar la respuesta a la quimioterapia.

Funciones
---------
dice_score           -- coeficiente Dice entre dos máscaras binarias
compute_brats_metrics -- calcula WT/TC/ET Dice para una predicción
"""

import torch


def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Coeficiente Dice entre dos máscaras binarias.

    Dice = (2 · |A ∩ B| + ε) / (|A| + |B| + ε)

    Valores: 0 = sin solapamiento, 1 = solapamiento perfecto.
    El término ε evita división por cero cuando ambas máscaras están vacías.

    Parameters
    ----------
    pred_mask : tensor booleano de predicción
    true_mask : tensor booleano de ground truth
    smooth    : término de suavizado ε

    Returns
    -------
    float : coeficiente Dice en [0, 1]
    """
    inter = (pred_mask & true_mask).sum().float()
    union = pred_mask.sum().float() + true_mask.sum().float()
    return ((2 * inter + smooth) / (union + smooth)).item()


def compute_brats_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """
    Calcula las métricas Dice estándar del benchmark BraTS.

    Acepta predicciones 2D (H, W) o 3D (H, W, D) con etiquetas {0,1,2,3}.

    Parameters
    ----------
    pred : tensor con etiquetas predichas
    gt   : tensor con etiquetas de ground truth

    Returns
    -------
    dict con claves "WT", "TC", "ET" y valores float en [0, 1]
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)

    return {
        "WT": dice_score(pred > 0,               gt > 0),
        "TC": dice_score((pred == 1) | (pred == 3), (gt == 1) | (gt == 3)),
        "ET": dice_score(pred == 3,              gt == 3),
    }
