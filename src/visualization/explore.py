"""
src/visualization/explore.py
============================
Funciones de exploración visual del dataset BraTS 2021.

Funciones
---------
make_seg_cmap    -- crea el colormap estándar para máscaras de segmentación
plot_modalities  -- muestra las 4 modalidades + segmentación en vista axial
plot_histograms  -- histogramas de intensidad por modalidad
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


# Etiquetas del colormap de segmentación
SEG_LABELS  = ["Necrosis (1)", "Edema (2)", "Tumor realzado (3)"]
SEG_COLORS  = ["#3A86FF", "#8BC34A", "#FF3A3A"]
MODALITIES  = ["flair", "t1", "t1ce", "t2"]


def make_seg_cmap() -> ListedColormap:
    """
    Devuelve el colormap de cuatro colores para máscaras de segmentación BraTS.

    Clase 0 (fondo) → transparente
    Clase 1 (necrosis)        → azul   (#3A86FF)
    Clase 2 (edema)           → verde  (#8BC34A)
    Clase 3 (tumor realzado)  → rojo   (#FF3A3A)
    """
    return ListedColormap([(0, 0, 0, 0)] + SEG_COLORS)


def _best_slice(vol: np.ndarray, axis: int) -> int:
    """
    Devuelve el índice del corte con más vóxeles de cerebro (> 0) a lo
    largo de `axis`. Útil para encontrar automáticamente el corte central
    del cerebro aunque el paciente no esté perfectamente centrado.
    """
    other_axes = tuple(i for i in range(vol.ndim) if i != axis)
    counts = (vol > 0).sum(axis=other_axes)
    return int(np.argmax(counts))


def plot_modalities(
    patient_id: str,
    base_dir: str | Path,
    slice_z: int | None = None,
    slice_x: int | None = None,  # mantenido por compatibilidad, no se usa
) -> None:
    """
    Visualiza las cuatro modalidades de MRI y la segmentación en vista axial.

    Cuando `slice_z` es None (por defecto) se detecta automáticamente el
    corte axial con más tejido cerebral en el volumen FLAIR.

    Parameters
    ----------
    patient_id : ID del paciente (sin prefijo BraTS2021_)
    base_dir   : directorio raíz con las carpetas BraTS2021_*/
    slice_z    : índice del corte axial (None → auto-detect desde FLAIR)
    slice_x    : ignorado; mantenido por compatibilidad con llamadas anteriores
    """
    base_dir = Path(base_dir)
    folder   = base_dir / f"BraTS2021_{patient_id}"
    seg_cmap = make_seg_cmap()

    # Cargar FLAIR primero — se usa para auto-detectar el corte axial y
    # como primera columna en la figura (evita una segunda carga desde disco)
    flair_vol = nib.load(folder / f"BraTS2021_{patient_id}_flair.nii.gz").get_fdata()

    if slice_z is None:
        slice_z = _best_slice(flair_vol, axis=2)  # eje Z → axial

    # Cargar y reasignar segmentación (4 → 3)
    seg_vol = nib.load(folder / f"BraTS2021_{patient_id}_seg.nii.gz").get_fdata().astype(int)
    seg_vol[seg_vol == 4] = 3

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f"BraTS2021_{patient_id} — Vista axial  z={slice_z}",
        fontsize=14, fontweight="bold",
    )

    for col, mod in enumerate(MODALITIES):
        # Reutiliza el FLAIR ya cargado; carga el resto desde disco
        vol = flair_vol if mod == "flair" else \
              nib.load(folder / f"BraTS2021_{patient_id}_{mod}.nii.gz").get_fdata()

        axes[col].imshow(vol[:, :, slice_z].T, cmap="bone", origin="lower")
        axes[col].set_title(mod.upper())
        axes[col].axis("off")

    # Segmentación axial
    axes[4].imshow(np.zeros_like(seg_vol[:, :, slice_z].T), cmap="bone", origin="lower")
    axes[4].imshow(
        seg_vol[:, :, slice_z].T,
        cmap=seg_cmap, vmin=0, vmax=3, origin="lower", alpha=0.9,
    )
    axes[4].set_title("Segmentación")
    axes[4].axis("off")

    patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(SEG_COLORS, SEG_LABELS)
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=3,
        fontsize=10, bbox_to_anchor=(0.5, -0.08),
    )
    plt.tight_layout()
    plt.show()


def plot_histograms(patient_id: str, base_dir: str | Path) -> None:
    """
    Muestra los histogramas de intensidad de las cuatro modalidades.

    Solo considera vóxeles de cerebro (intensidad > 0) para excluir
    el fondo negro que dominaría la distribución.

    Parameters
    ----------
    patient_id : ID del paciente (sin prefijo BraTS2021_)
    base_dir   : directorio raíz con las carpetas BraTS2021_*/
    """
    base_dir = Path(base_dir)
    folder   = base_dir / f"BraTS2021_{patient_id}"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Histogramas de intensidad — BraTS2021_{patient_id}",
        fontsize=13, fontweight="bold",
    )

    for ax, mod in zip(axes, MODALITIES):
        vol   = nib.load(folder / f"BraTS2021_{patient_id}_{mod}.nii.gz").get_fdata()
        brain = vol[vol > 0]
        ax.hist(brain, bins=100, color="#4361ee", alpha=0.75, edgecolor="none")
        ax.set_title(mod.upper())
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()
