"""
src/training/trainer.py
=======================
Funciones de entrenamiento y validación con soporte AMP (float16).

Funciones
---------
train_one_epoch -- una pasada completa de entrenamiento sobre el loader
validate_epoch  -- evaluación sobre el set de validación (sin gradientes)
train_model     -- bucle completo de entrenamiento con guardado del mejor modelo
"""

import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.dataset import BraTSCachedDataset, BraTSFastDataset
from src.training.loss import CombinedLoss


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> float:
    """
    Ejecuta una época completa de entrenamiento.

    Usa Automatic Mixed Precision (AMP) cuando se proporciona un GradScaler.
    AMP convierte automáticamente partes del forward pass a float16,
    reduciendo el uso de VRAM y acelerando el cómputo en GPUs modernas
    sin pérdida significativa de precisión.

    Parameters
    ----------
    model     : modelo en modo train
    loader    : DataLoader del split de entrenamiento
    optimizer : optimizador (Adam, AdamW, etc.)
    loss_fn   : función de pérdida
    device    : dispositivo de cómputo
    scaler    : GradScaler para AMP; None para entrenamiento en float32

    Returns
    -------
    float : pérdida media por batch en esta época
    """
    model.train()
    total = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                loss = loss_fn(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()

        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Evalúa el modelo sobre el set de validación.

    La validación se realiza en float32 (sin AMP) para evitar valores
    NaN que pueden aparecer en DiceLoss cuando los logits son muy pequeños
    en precisión float16.

    Parameters
    ----------
    model   : modelo en modo eval
    loader  : DataLoader del split de validación
    loss_fn : función de pérdida
    device  : dispositivo de cómputo

    Returns
    -------
    float : pérdida media de validación
    """
    model.eval()
    total = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        loss   = loss_fn(model(images), masks)
        total += loss.item()

    return total / len(loader)


def train_model(
    model: torch.nn.Module,
    train_ids: list,
    val_ids: list,
    cache_dir: str | Path,
    device: torch.device,
    num_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: str = "best_unet_brats.pth",
) -> dict:
    """
    Bucle principal de entrenamiento del U-Net.

    Selección automática de dataset:
    - Si existen {cache_dir}/train_X.npy y val_X.npy → BraTSFastDataset
    - Si no existen → BraTSCachedDataset (fallback, más lento)

    Estrategia de guardado del modelo:
    - Guarda cuando val_loss mejora (es menor que el mejor hasta ahora)
    - Si val_loss es NaN (puede ocurrir en primeras épocas), usa train_loss
    - Siempre guarda en la última época como seguridad

    Hiperparámetros adicionales:
    - ReduceLROnPlateau: reduce lr × 0.5 si val_loss no mejora en 3 épocas
    - cudnn.benchmark=True: optimiza operaciones para tamaños fijos de entrada

    Parameters
    ----------
    model       : UNet (u otro modelo compatible)
    train_ids   : lista de IDs de paciente para entrenamiento
    val_ids     : lista de IDs de paciente para validación
    cache_dir   : directorio del caché de cortes 2D
    device      : dispositivo de cómputo (cuda / cpu)
    num_epochs  : número de épocas
    batch_size  : tamaño de batch
    lr          : learning rate inicial
    save_path   : ruta donde guardar el mejor modelo

    Returns
    -------
    dict con claves "train_loss" y "val_loss" (listas por época)
    """
    torch.backends.cudnn.benchmark = True
    cache_dir    = Path(cache_dir)
    train_prefix = cache_dir / "train"
    val_prefix   = cache_dir / "val"

    # Selección de dataset
    fast_ok = (
        Path(str(train_prefix) + "_X.npy").exists()
        and Path(str(val_prefix) + "_X.npy").exists()
    )
    if fast_ok:
        print("✅ Usando BraTSFastDataset (arrays consolidados)")
        train_ds = BraTSFastDataset(train_prefix)
        val_ds   = BraTSFastDataset(val_prefix)
    else:
        print("⚠️  Arrays consolidados no encontrados — usando BraTSCachedDataset")
        print("   Ejecuta consolidate_cache() para reducir el tiempo por época ~6×.")
        train_ds = BraTSCachedDataset(train_ids, cache_dir)
        val_ds   = BraTSCachedDataset(val_ids,   cache_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    loss_fn = CombinedLoss(num_classes=4).to(device)
    scaler  = GradScaler("cuda") if device.type == "cuda" else None

    history    = {"train_loss": [], "val_loss": []}
    best_loss  = float("inf")
    best_epoch = -1

    print(f"\nInicio: {num_epochs} épocas | batch={batch_size} | "
          f"train={len(train_ds)} val={len(val_ds)} muestras\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tr_loss  = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        is_nan  = val_loss != val_loss          # True si NaN
        metric  = val_loss if not is_nan else tr_loss
        if not is_nan:
            scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        is_last = (epoch == num_epochs)
        is_best = (metric < best_loss)
        flag    = ""

        if is_best or is_last:
            if is_best:
                best_loss  = metric
                best_epoch = epoch
                flag = "  ← mejor guardado"
            elif is_last:
                flag = "  ← última época (guardado)"
            torch.save(model.state_dict(), save_path)

        val_str = f"{val_loss:.4f}" if not is_nan else "nan(⚠)"
        elapsed = time.time() - t0
        print(f"Época {epoch:02d}/{num_epochs}  train={tr_loss:.4f}  "
              f"val={val_str}  [{elapsed:.0f}s]{flag}")

    print(f"\n✅ Entrenamiento completado.")
    print(f"   Mejor época: {best_epoch}  (loss={best_loss:.4f})")
    print(f"   Modelo guardado en: {save_path}")
    return history
