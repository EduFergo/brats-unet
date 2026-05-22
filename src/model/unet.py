"""
src/model/unet.py
=================
Arquitectura U-Net 2D para segmentación semántica de imágenes médicas.

Clases
------
DoubleConv    -- bloque Conv→BN→ReLU→Conv→BN→ReLU
EncoderBlock  -- DoubleConv + MaxPool (encoder path)
DecoderBlock  -- ConvTranspose + concatenación de skip + DoubleConv (decoder path)
UNet          -- red completa encoder–bottleneck–decoder

Referencia
----------
Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.
https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Bloques básicos ───────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Bloque de dos convoluciones 3×3 con Batch Normalization y ReLU.

    Estructura:
        Conv2d(3×3, padding=1) → BatchNorm2d → ReLU
        Conv2d(3×3, padding=1) → BatchNorm2d → ReLU

    El padding=1 conserva las dimensiones espaciales (H, W), y
    bias=False porque BatchNorm ya incluye un término de sesgo.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Bloque del camino de codificación (encoder path / contraction path).

    Aplica DoubleConv y guarda el resultado como conexión de salto (skip
    connection) antes de aplicar MaxPool2d(2) para reducir la resolución
    espacial a la mitad.

    Returns
    -------
    (x_down, skip) : tensor reducido y tensor de skip connection
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """
    Bloque del camino de decodificación (decoder path / expansion path).

    Pasos:
    1. ConvTranspose2d × 2 → sube la resolución espacial al doble
    2. Concatenación con el skip connection del encoder (misma resolución)
    3. DoubleConv para fusionar la información local y global

    Si hay discrepancia de tamaño por redondeos, se aplica interpolación
    bilineal para alinear tensores antes de la concatenación.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ── Red completa ──────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net 2D para segmentación semántica multi-clase.

    La arquitectura en U tiene dos caminos simétricos:

    Encoder (contraction path):
        Reduce resolución espacial (÷2 en cada nivel) y aumenta canales,
        capturando contexto semántico de alto nivel.

    Decoder (expansion path):
        Recupera resolución espacial y combina con las skip connections
        del encoder para preservar detalles espaciales finos.

    Las skip connections son la innovación clave de U-Net: evitan la
    pérdida de información de localización que ocurriría si solo hubiera
    un bottleneck.

    Parameters
    ----------
    in_channels : número de modalidades de entrada (4 en BraTS)
    num_classes : número de clases de salida (4: fondo, necrosis, edema, ET)
    features    : canales en cada nivel del encoder [32, 64, 128, 256]

    Parámetros con features=[32,64,128,256]: ~7.8M (apto para 6 GB VRAM)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        features: list = [32, 64, 128, 256],
    ):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # Bottleneck (nivel más profundo, sin pooling)
        self.bottleneck = DoubleConv(ch, ch * 2)
        ch = ch * 2

        # Decoder
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(ch, f, f))
            ch = f

        # Cabeza de clasificación 1×1
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)  # logits (B, num_classes, H, W)
