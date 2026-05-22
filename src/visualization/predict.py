"""
src/visualization/predict.py
============================
Visualización de predicciones del modelo, post-procesamiento 3D y evaluación.

Funciones
---------
plot_history         -- curvas de pérdida train / val por época
visualize_prediction -- visualización 2D de una muestra (GT vs predicción)
postprocess_3d       -- filtrado de componentes conexas 3D para eliminar islas
predict_patient_3d   -- inferencia 2D → reconstrucción de volumen 3D
                        carga el T2 completo (155 cortes) para visualización
                        sin recortes; expande pred/GT al mismo tamaño
visualize_patient_3d -- vista multiplanar 2 columnas: GT | Predicción 2D
                        fondo T2 completo → cerebro sin "partidos"
visualize_tumor_3d   -- nube de puntos 3D interactiva (Plotly): predicción
                        + GT semitransparente para comparar
evaluate_val_set     -- Dice WT/TC/ET promedio sobre el set de validación
"""

import statistics
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from src.training.metrics import compute_brats_metrics
from src.visualization.explore import SEG_COLORS, SEG_LABELS, make_seg_cmap


# ── Historial de entrenamiento ────────────────────────────────────────────────

def plot_history(history: dict) -> None:
    """
    Dibuja las curvas de pérdida de entrenamiento y validación por época.

    Parameters
    ----------
    history : dict con claves "train_loss" y "val_loss" (listas de floats)
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    val_ok = [v for v in history["val_loss"] if v == v]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], "b-o", label="Train loss")
    if val_ok:
        ax.plot(list(epochs)[: len(val_ok)], val_ok, "r-o", label="Val loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.set_title("Historial de entrenamiento", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Visualización 2D (por corte individual) ───────────────────────────────────

@torch.no_grad()
def visualize_prediction(
    model: torch.nn.Module,
    dataset,
    idx: int,
    device: torch.device,
) -> None:
    """
    Muestra la predicción del modelo sobre un corte 2D individual.

    Panel de 4 columnas:
    1. T2 de entrada
    2. Ground truth
    3. Predicción del modelo
    4. Superposición sobre T2

    Parameters
    ----------
    model   : UNet en modo eval
    dataset : BraTSCachedDataset o BraTSFastDataset
    idx     : índice de la muestra
    device  : dispositivo de cómputo
    """
    seg_cmap = make_seg_cmap()
    model.eval()

    x, y   = dataset[idx]
    logits = model(x.unsqueeze(0).to(device))
    pred   = logits.argmax(dim=1).squeeze().cpu()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Predicción 2D — Muestra #{idx}", fontsize=14, fontweight="bold")

    axes[0].imshow(x[3].T, cmap="bone", origin="lower")
    axes[0].set_title("T2 (entrada)")

    axes[1].imshow(y.T, cmap=seg_cmap, vmin=0, vmax=3, origin="lower")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred.T, cmap=seg_cmap, vmin=0, vmax=3, origin="lower")
    axes[2].set_title("Predicción 2D")

    axes[3].imshow(x[3].T, cmap="bone", origin="lower")
    ov = pred.T.float()
    ov[ov == 0] = float("nan")
    axes[3].imshow(ov, cmap=seg_cmap, vmin=0, vmax=3, origin="lower", alpha=0.6)
    axes[3].set_title("Superposición")

    m = compute_brats_metrics(pred, y)
    for ax in axes:
        ax.axis("off")
    plt.figtext(
        0.5, -0.02,
        f"Dice → WT: {m['WT']:.3f}  |  TC: {m['TC']:.3f}  |  ET: {m['ET']:.3f}",
        ha="center", fontsize=12,
    )
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(SEG_COLORS, SEG_LABELS)]
    fig.legend(
        handles=patches, loc="lower center", ncol=3,
        fontsize=10, bbox_to_anchor=(0.5, -0.08),
    )
    plt.tight_layout()
    plt.show()


# ── Post-procesamiento 3D ─────────────────────────────────────────────────────

def postprocess_3d(pred_vol: np.ndarray, min_voxels: int = 100) -> np.ndarray:
    """
    Elimina componentes conexas 3D pequeñas por clase.

    El U-Net 2D predice cada corte axial de forma independiente. Al apilar
    los cortes aparecen "islas" de predicción aisladas (falsos positivos sin
    continuidad espacial). Este post-procesamiento aplica análisis de
    componentes conexas 3D (scipy.ndimage.label) y descarta los componentes
    con menos de `min_voxels` vóxeles.

    Parameters
    ----------
    pred_vol   : np.ndarray (H, W, D) con etiquetas {0,1,2,3}
    min_voxels : umbral mínimo de vóxeles para conservar un componente

    Returns
    -------
    np.ndarray (H, W, D) sin islas pequeñas
    """
    result = np.zeros_like(pred_vol)
    for cls in [1, 2, 3]:
        binary        = pred_vol == cls
        labeled, n_cc = ndimage.label(binary)
        for comp_id in range(1, n_cc + 1):
            mask = labeled == comp_id
            if mask.sum() >= min_voxels:
                result[mask] = cls
    return result


# ── Carga de T2 completo desde NIfTI ─────────────────────────────────────────

def _load_t2_full(
    patient_id: str,
    base_dir: str | Path,
    img_size: int,
) -> tuple[np.ndarray, int]:
    """
    Carga el volumen T2 completo (todos los cortes z del NIfTI).

    Redimensiona cada corte a img_size×img_size y normaliza con el
    percentil 99 de los vóxeles de cerebro (intensidad > 0) para
    visualización. Devuelve None si el archivo no existe.

    Returns
    -------
    t2_vol : np.ndarray (img_size, img_size, full_D) float32 en [0,1]
    full_D : número total de cortes z en el NIfTI original
    """
    t2_path = (
        Path(base_dir)
        / f"BraTS2021_{patient_id}"
        / f"BraTS2021_{patient_id}_t2.nii.gz"
    )
    if not t2_path.exists():
        return None, 0

    t2_data = nib.load(t2_path).get_fdata().astype(np.float32)
    full_D  = t2_data.shape[2]  # = 155 en BraTS 2021

    t2_vol = np.zeros((img_size, img_size, full_D), dtype=np.float32)
    for z in range(full_D):
        t2_vol[:, :, z] = cv2.resize(
            t2_data[:, :, z], (img_size, img_size),
            interpolation=cv2.INTER_LINEAR,
        )

    # Normalización por percentil-99 para evitar que hot-spots saturen
    brain = t2_vol[t2_vol > 0]
    if len(brain):
        v_max = float(np.percentile(brain, 99))
        t2_vol = np.clip(t2_vol / (v_max + 1e-8), 0.0, 1.0)

    return t2_vol, full_D


# ── Inferencia y reconstrucción 3D ────────────────────────────────────────────

@torch.no_grad()
def predict_patient_3d(
    model: torch.nn.Module,
    patient_id: str,
    cache_dir: str | Path,
    device: torch.device,
    base_dir: str | Path = None,
    img_size: int = 128,
    z_min: int = 10,
    z_max: int = 145,
    min_voxels: int = 100,
):
    """
    Reconstruye el volumen de predicción 3D para un paciente completo.

    Proceso:
    1. Lee todos los cortes .npy cacheados del paciente (z_min a z_max)
    2. Ejecuta inferencia 2D en cada corte
    3. Apila en volumen (H, W, D_partial)
    4. Aplica postprocess_3d() para eliminar islas
    5. Si se pasa base_dir, carga el T2 NIfTI COMPLETO (todos los z)
       y expande pred/GT al mismo tamaño (D_full), con ceros fuera del
       rango cacheado. Esto evita el "cerebro partido" en vistas
       coronal y sagital.

    Parameters
    ----------
    model      : UNet en modo eval
    patient_id : ID del paciente (sin prefijo BraTS2021_)
    cache_dir  : directorio con los .npy del caché
    device     : dispositivo de cómputo
    base_dir   : directorio de datos NIfTI — activa la carga de T2 completo
    img_size   : resolución espacial de los cortes (H = W)
    z_min      : primer corte axial del rango cacheado
    z_max      : último corte axial (exclusivo)
    min_voxels : umbral para postprocess_3d

    Returns
    -------
    pred_raw : np.ndarray (H, W, D) predicción 2D sin post-procesar
    pred_pp  : np.ndarray (H, W, D) predicción post-procesada
    gt_vol   : np.ndarray (H, W, D) ground truth
    z_indices: lista de índices z con datos en el caché
    t2_vol   : np.ndarray (H, W, D) T2 normalizado, o None

    Nota: si se pasa base_dir, D = full_D (155 cortes del NIfTI completo)
    y las predicciones están embebidas en las posiciones z_min:z_max.
    Si no se pasa base_dir, D = z_max - z_min = 135.
    """
    cache_dir = Path(cache_dir)
    model.eval()

    H = W = img_size
    D_partial = z_max - z_min
    pred_raw  = np.zeros((H, W, D_partial), dtype=np.int64)
    gt_vol    = np.zeros((H, W, D_partial), dtype=np.int64)
    z_indices = []

    slice_files = sorted(cache_dir.glob(f"{patient_id}_z*_x.npy"))
    if not slice_files:
        raise FileNotFoundError(
            f"No se encontraron cortes en caché para paciente {patient_id}"
        )

    for f in slice_files:
        z_str = f.stem.split("_z")[1].replace("_x", "")
        z_idx = int(z_str)
        if z_idx < z_min or z_idx >= z_max:
            continue

        x = torch.from_numpy(np.load(f)).unsqueeze(0).to(device)
        y = np.load(str(f).replace("_x.npy", "_y.npy")).astype(np.int64)

        pred = model(x).argmax(dim=1).squeeze().cpu().numpy()

        d = z_idx - z_min
        pred_raw[:, :, d] = pred
        gt_vol  [:, :, d] = y
        z_indices.append(z_idx)

    pred_pp = postprocess_3d(pred_raw, min_voxels)

    tumor_raw = int((pred_raw > 0).sum())
    tumor_pp  = int((pred_pp  > 0).sum())
    print(
        f"Paciente {patient_id}: {len(z_indices)} cortes | "
        f"vóxeles tumor  raw={tumor_raw} → pp={tumor_pp} "
        f"(−{tumor_raw - tumor_pp} eliminados por post-proc)"
    )

    # ── Cargar T2 completo y expandir volúmenes a D_full ─────────────────────
    if base_dir is None:
        return pred_raw, pred_pp, gt_vol, z_indices, None

    t2_vol, full_D = _load_t2_full(patient_id, base_dir, img_size)

    if t2_vol is None or full_D == 0:
        return pred_raw, pred_pp, gt_vol, z_indices, None

    # Embeber las predicciones parciales dentro del volumen completo
    pred_raw_full = np.zeros((H, W, full_D), dtype=np.int64)
    pred_pp_full  = np.zeros((H, W, full_D), dtype=np.int64)
    gt_vol_full   = np.zeros((H, W, full_D), dtype=np.int64)

    pred_raw_full[:, :, z_min:z_max] = pred_raw
    pred_pp_full [:, :, z_min:z_max] = pred_pp
    gt_vol_full  [:, :, z_min:z_max] = gt_vol

    return pred_raw_full, pred_pp_full, gt_vol_full, z_indices, t2_vol


# ── Vista multiplanar 2D (axial · coronal · sagital) ─────────────────────────

def visualize_patient_3d(
    pred_raw: np.ndarray,
    gt_vol: np.ndarray,
    patient_id: str = "",
    t2_vol: np.ndarray = None,
) -> dict:
    """
    Visualización de la predicción 2D frente al ground truth en tres cortes axiales.

    Muestra dos filas (GT arriba, Predicción abajo) × tres columnas
    (tres cortes axiales representativos del tumor). Todos los cortes
    son axiales para garantizar la misma orientación del cerebro.

    Parameters
    ----------
    pred_raw   : np.ndarray (H, W, D) — predicción 2D (sin post-proc)
    gt_vol     : np.ndarray (H, W, D) — ground truth
    patient_id : string para el título
    t2_vol     : np.ndarray (H, W, D) en [0,1] — fondo T2 completo

    Returns
    -------
    dict con métricas Dice 3D {"WT", "TC", "ET"}
    """
    seg_cmap = make_seg_cmap()
    H, W, D  = pred_raw.shape

    # Seleccionar 3 cortes axiales representativos del tumor
    tumor_zs = sorted([d for d in range(D) if (gt_vol[:, :, d] > 0).sum() > 10])
    if len(tumor_zs) >= 3:
        n = len(tumor_zs)
        slices = [tumor_zs[n // 5], tumor_zs[n // 2], tumor_zs[4 * n // 5]]
    else:
        slices = [max(0, D // 4), D // 2, min(D - 1, 3 * D // 4)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Paciente {patient_id} — Ground Truth vs Predicción 2D",
        fontsize=14, fontweight="bold",
    )

    row_labels = ["Ground Truth", "Predicción 2D (U-Net)"]
    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=11, fontweight="600", labelpad=10)

    for col, z in enumerate(slices):
        axes[0, col].set_title(f"z = {z}", fontsize=10)

        for row, seg_data in enumerate([gt_vol, pred_raw]):
            ax = axes[row, col]

            # Fondo T2 — muestra toda la anatomía cerebral
            if t2_vol is not None:
                ax.imshow(t2_vol[:, :, z].T, cmap="bone", origin="lower", vmin=0, vmax=1)
            else:
                ax.imshow(np.zeros((H, W)), cmap="bone", origin="lower")

            # Superposición de segmentación (fondo transparente)
            seg_slc = seg_data[:, :, z].T.astype(float)
            seg_slc[seg_slc == 0] = float("nan")
            ax.imshow(seg_slc, cmap=seg_cmap, vmin=0, vmax=3, origin="lower", alpha=0.75)
            ax.axis("off")

    # Métricas 3D sobre el volumen completo
    m = compute_brats_metrics(
        torch.tensor(pred_raw.ravel()), torch.tensor(gt_vol.ravel())
    )
    plt.figtext(
        0.5, -0.01,
        f"Dice 3D — WT: {m['WT']:.3f}  |  TC: {m['TC']:.3f}  |  ET: {m['ET']:.3f}",
        ha="center", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor="#aab4ff"),
    )
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(SEG_COLORS, SEG_LABELS)]
    fig.legend(
        handles=patches, loc="lower center", ncol=3,
        fontsize=10, bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    plt.show()
    return m


# ── Visualización 3D interactiva (Plotly) ────────────────────────────────────

def visualize_tumor_3d(
    pred_pp: np.ndarray,
    gt_vol: np.ndarray = None,
    patient_id: str = "",
    max_points: int = 15_000,
) -> None:
    """
    Visualización 3D interactiva del tumor con Plotly — GT y Predicción lado a lado.

    Muestra dos subplots 3D en paralelo: Ground Truth (izquierda) y
    Predicción post-procesada (derecha). Misma escala de colores en ambos.

    Convención de colores:
      Azul  (#3A86FF) → Necrosis (clase 1)
      Verde (#8BC34A) → Edema peritumoral (clase 2)
      Rojo  (#FF3A3A) → Tumor realzado / ET (clase 3)

    Parameters
    ----------
    pred_pp    : np.ndarray (H, W, D) — predicción post-procesada {0,1,2,3}
    gt_vol     : np.ndarray (H, W, D) — ground truth (opcional)
    patient_id : string para el título
    max_points : máximo de puntos por clase (submuestreo si se supera)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly no instalado. Ejecuta: uv sync")
        return

    cls_config = [
        (1, "#3A86FF", "Necrosis"),
        (2, "#8BC34A", "Edema peritumoral"),
        (3, "#FF3A3A", "Tumor realzado (ET)"),
    ]
    rng = np.random.default_rng(42)

    def _make_scatter(vol, cls, color, label, show_legend):
        xs, ys, zs = np.where(vol == cls)
        n = len(xs)
        if n == 0:
            return None
        if n > max_points:
            idx = rng.choice(n, max_points, replace=False)
            xs, ys, zs = xs[idx], ys[idx], zs[idx]
        return go.Scatter3d(
            x=xs.tolist(), y=ys.tolist(), z=zs.tolist(),
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.7, line=dict(width=0)),
            name=label,
            legendgroup=label,
            showlegend=show_legend,
        )

    has_gt   = gt_vol is not None
    n_cols   = 2 if has_gt else 1
    col_titles = ["Ground Truth", "Predicción U-Net"] if has_gt else ["Predicción U-Net"]

    fig = make_subplots(
        rows=1, cols=n_cols,
        specs=[[{"type": "scatter3d"}] * n_cols],
        subplot_titles=col_titles,
        horizontal_spacing=0.02,
    )

    # GT traces — leyenda mostrada aquí (col 1)
    if has_gt:
        for cls, color, label in cls_config:
            t = _make_scatter(gt_vol, cls, color, label, show_legend=True)
            if t is not None:
                fig.add_trace(t, row=1, col=1)

    # Pred traces — misma leyenda compartida (sin duplicar)
    pred_col = 2 if has_gt else 1
    for cls, color, label in cls_config:
        t = _make_scatter(pred_pp, cls, color, label, show_legend=(not has_gt))
        if t is not None:
            fig.add_trace(t, row=1, col=pred_col)

    if not fig.data:
        print("No se encontraron vóxeles de tumor.")
        return

    scene_cfg = dict(
        xaxis=dict(title="Sagital (H)", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Coronal (W)", showgrid=True, gridcolor="#e0e0e0"),
        zaxis=dict(title="Axial (D)",   showgrid=True, gridcolor="#e0e0e0"),
        bgcolor="white",
        aspectmode="data",
    )

    layout_kw = dict(
        title=dict(
            text=f"Segmentación 3D — Paciente {patient_id}",
            font=dict(size=15), x=0.5,
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            tracegroupgap=4,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=620,
        scene=scene_cfg,
    )
    if has_gt:
        layout_kw["scene2"] = scene_cfg

    fig.update_layout(**layout_kw)
    fig.show()


# ── Evaluación del set de validación ─────────────────────────────────────────

@torch.no_grad()
def evaluate_val_set(
    model: torch.nn.Module,
    val_ids: list,
    cache_dir: str | Path,
    device: torch.device,
    min_voxels: int = 100,
    n_patients: int = None,
) -> list:
    """
    Calcula Dice WT/TC/ET promedio sobre el set de validación completo.

    Compara predicción raw (2D) vs post-procesada (3D) para cada paciente.

    Parameters
    ----------
    model      : UNet entrenado
    val_ids    : lista de IDs del set de validación
    cache_dir  : directorio del caché
    device     : dispositivo de cómputo
    min_voxels : umbral para postprocess_3d
    n_patients : si se especifica, evalúa solo los primeros N pacientes

    Returns
    -------
    list de dicts con métricas por paciente
    """
    ids  = val_ids[:n_patients] if n_patients else val_ids
    rows = []

    for pid in tqdm(ids, desc="Evaluando pacientes"):
        try:
            raw, pp, gt, _, _ = predict_patient_3d(
                model, pid, cache_dir, device, min_voxels=min_voxels
            )
            m_r = compute_brats_metrics(torch.tensor(raw.ravel()), torch.tensor(gt.ravel()))
            m_p = compute_brats_metrics(torch.tensor(pp.ravel()),  torch.tensor(gt.ravel()))
            rows.append({
                "pid":    pid,
                "WT_raw": m_r["WT"], "TC_raw": m_r["TC"], "ET_raw": m_r["ET"],
                "WT_pp":  m_p["WT"], "TC_pp":  m_p["TC"], "ET_pp":  m_p["ET"],
            })
        except FileNotFoundError:
            continue

    if not rows:
        print("Sin resultados — verifica que el caché esté disponible.")
        return []

    print(f"\nResultados sobre {len(rows)} pacientes:")
    print(f"{'Versión':<12}  {'WT':>6}  {'TC':>6}  {'ET':>6}")
    print("-" * 35)
    for split, suffix in [("RAW (2D)", "_raw"), ("POST (3D)", "_pp")]:
        wt = statistics.mean(r[f"WT{suffix}"] for r in rows)
        tc = statistics.mean(r[f"TC{suffix}"] for r in rows)
        et = statistics.mean(r[f"ET{suffix}"] for r in rows)
        print(f"{split:<12}  {wt:>6.3f}  {tc:>6.3f}  {et:>6.3f}")

    return rows
