# 🧠 Segmentación de Tumores Cerebrales con U-Net (BraTS 2021)

Pipeline académico completo de segmentación semántica de gliomas cerebrales
usando una **U-Net 2D** entrenada sobre el dataset **BraTS 2021**.
Incluye preprocesado, entrenamiento con AMP, post-procesamiento 3D y
visualización interactiva con Plotly.

---

## Resultados

El modelo predice tres regiones tumorales superpuestas según el estándar BraTS:

| Región | Clases | Descripción |
|---|---|---|
| **WT** — Whole Tumour | 1 + 2 + 3 | Toda la lesión |
| **TC** — Tumour Core | 1 + 3 | Núcleo (resección quirúrgica) |
| **ET** — Enhancing Tumour | 3 | Zona de realce con contraste |

La métrica de evaluación es el **coeficiente Dice** (0 = sin superposición, 1 = perfecto).

---

## Estructura del proyecto

```
brats_unet/
├── brats_unet_tutorial.ipynb   ← notebook principal (pipeline completo)
├── pyproject.toml              ← dependencias gestionadas con uv
└── src/
    ├── data/
    │   ├── preprocessing.py    ← zscore_normalize, build_cache
    │   └── dataset.py          ← BraTSCachedDataset, BraTSFastDataset, consolidate_cache
    ├── model/
    │   └── unet.py             ← arquitectura U-Net 2D con skip connections
    ├── training/
    │   ├── loss.py             ← DiceLoss, CombinedLoss (Dice + CrossEntropy)
    │   ├── metrics.py          ← compute_brats_metrics (Dice WT/TC/ET)
    │   └── trainer.py          ← entrenamiento con AMP + GradScaler + ReduceLROnPlateau
    └── visualization/
        ├── explore.py          ← plot_modalities, plot_histograms
        └── predict.py          ← predict_patient_3d, visualize_patient_3d,
                                   visualize_tumor_3d, evaluate_val_set
```

---

## Requisitos

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (gestor de dependencias)
- GPU con CUDA recomendada (funciona en CPU pero lento)

---

## Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/EduFergo/brats-unet.git
cd brats-unet

# 2. Instala las dependencias con uv
uv sync
```

Las dependencias principales son: `torch`, `nibabel`, `numpy`, `matplotlib`,
`opencv-python`, `scipy`, `plotly` y `tqdm`.

---

## Dataset

El proyecto usa el dataset **BraTS 2021 Task 1** (~100 GB descomprimido).

1. Descárgalo desde Kaggle:
   [dschettler8845/brats-2021-task1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

2. Extrae el contenido en `data/` dentro de la carpeta del proyecto:
   ```
   brats_unet/
   └── data/
       ├── BraTS2021_00000/
       │   ├── BraTS2021_00000_flair.nii.gz
       │   ├── BraTS2021_00000_t1.nii.gz
       │   ├── BraTS2021_00000_t1ce.nii.gz
       │   ├── BraTS2021_00000_t2.nii.gz
       │   └── BraTS2021_00000_seg.nii.gz
       └── ...
   ```

La primera celda del notebook descarga el dataset automáticamente via
`kagglehub` si `data/` está vacía (requiere credenciales de Kaggle configuradas).

---

## Uso

Abre y ejecuta `brats_unet_tutorial.ipynb` en orden.
El notebook está organizado en 20 secciones:

1. Entorno y dependencias
2. Imports
3. Configuración global (rutas, hiperparámetros)
4. Descarga del dataset
5. Exploración de archivos NIfTI
6. Visualización de modalidades y segmentación
7. Distribución de intensidades
8. Preprocesado: construcción del caché 2D
9. Dataset PyTorch
10. Consolidación del caché (optimización I/O)
11. Arquitectura U-Net 2D
12. Función de pérdida: Dice + Cross-Entropy
13. Métricas BraTS (Dice WT/TC/ET)
14. Entrenamiento
15. Curvas de pérdida
16. Predicciones 2D — inspección visual
17. Predicción 2D con fondo anatómico completo
18. Visualización 3D interactiva (Plotly)
19. Evaluación global del set de validación
20. Conclusiones

---

## Pipeline técnico

```
NIfTI 3D (240×240×155)
    ↓  Z-score normalización por vóxel de cerebro
    ↓  Resize a 128×128 por corte axial
    ↓  Filtrado de cortes sin tumor (< 200 vóxeles)
    ↓  Caché .npy en disco → memmap consolidado
    ↓  U-Net 2D (4 canales entrada, 4 clases salida)
    ↓  Pérdida: 0.5 × Dice + 0.5 × CrossEntropy
    ↓  Post-procesamiento: componentes conexas 3D
    ↓  Evaluación: Dice WT / TC / ET
```

**Entrada**: 4 modalidades MRI — FLAIR, T1, T1ce, T2  
**Salida**: máscara de segmentación — 0=fondo, 1=necrosis, 2=edema, 3=ET
