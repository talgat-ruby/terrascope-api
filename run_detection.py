"""
Car Detection — запуск на любом GeoTIFF снимке
Меняй IMAGE_PATH и CROP_CENTER_* чтобы тестировать разные зоны
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np, rasterio
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ─────────────────────────────────────────
#  НАСТРОЙКИ — меняй здесь
# ─────────────────────────────────────────

IMAGE_PATH = 'Dataset/Astana/Astana_6.tif'   # путь к снимку (любой тайл)

# Размер вырезаемой зоны в пикселях из оригинала (300м = 5555px при 5.4см/px)
CROP_SIZE = 5555

# Смещение центра зоны относительно центра снимка (0, 0 = точный центр)
OFFSET_X = 0   # пикселей вправо
OFFSET_Y = 0   # пикселей вниз

# Пороги confidence для сравнения
THRESHOLDS = [0.25, 0.15, 0.10]

# ─────────────────────────────────────────
#  ПУТИ (не менять)
# ─────────────────────────────────────────

from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,
    'terrascope-api/runs/detect/runs/car_detection/cowc_v4_cpu/weights/best.pt')

# Уникальное имя: имя снимка + смещение + время
_ts       = datetime.now().strftime('%H%M%S')
_imgname  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
_suffix   = f'dx{OFFSET_X}_dy{OFFSET_Y}' if (OFFSET_X or OFFSET_Y) else 'center'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_PATH  = os.path.join(RESULTS_DIR, f'{_imgname}__{_suffix}__{_ts}.png')
TMP_PATCH = os.path.join(BASE_DIR, '_tmp_patch.png')

OUR_GSD  = 0.054   # см/пикс наших снимков
COWC_GSD = 0.15    # GSD обучающего датасета
SCALE    = OUR_GSD / COWC_GSD   # 0.36

# ─────────────────────────────────────────
#  1. Читаем патч из снимка
# ─────────────────────────────────────────

print(f'\nСнимок: {IMAGE_PATH}')
with rasterio.open(os.path.join(BASE_DIR, IMAGE_PATH)) as src:
    H, W = src.height, src.width
    cx = W // 2 + OFFSET_X
    cy = H // 2 + OFFSET_Y
    crop = min(CROP_SIZE, cx, cy, W - cx, H - cy)
    x0, y0 = cx - crop // 2, cy - crop // 2
    win = rasterio.windows.Window(x0, y0, crop, crop)
    rgb = np.moveaxis(src.read([1, 2, 3], window=win), 0, -1).astype(np.uint8)

area_m = crop * OUR_GSD
print(f'Зона: {crop}×{crop} пикс = {area_m:.0f}м × {area_m:.0f}м')

# ─────────────────────────────────────────
#  2. Downsample до GSD обучения
# ─────────────────────────────────────────

new_size = int(crop * SCALE)
Image.fromarray(rgb).resize((new_size, new_size), Image.LANCZOS).save(TMP_PATCH)
print(f'После resize: {new_size}×{new_size} пикс (≈ COWC 15 см/пикс)\n')

# ─────────────────────────────────────────
#  3. SAHI inference для каждого порога
# ─────────────────────────────────────────

COLORS = {0.25: '#00ff88', 0.15: '#ffcc00', 0.10: '#ff6666'}
results = {}

for conf in THRESHOLDS:
    model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=MODEL_PATH,
        confidence_threshold=conf,
        device='cpu',
    )
    out = get_sliced_prediction(
        TMP_PATCH, model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        verbose=0,
    )
    results[conf] = out.object_prediction_list
    confs = [p.score.value for p in results[conf]]
    avg = sum(confs) / len(confs) if confs else 0
    print(f'  conf ≥ {conf}  →  {len(results[conf])} машин  |  avg confidence: {avg:.2f}')

# ─────────────────────────────────────────
#  4. Визуализация — 3 варианта рядом
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(30, 11))
fig.patch.set_facecolor('#0d0d0d')
inv = 1.0 / SCALE

img_name = os.path.basename(IMAGE_PATH)

for ax, conf in zip(axes, THRESHOLDS):
    preds = results[conf]
    ax.imshow(rgb)
    ax.set_title(
        f'conf ≥ {conf}   |   найдено: {len(preds)} машин',
        color=COLORS[conf], fontsize=14, fontweight='bold', pad=10
    )
    ax.axis('off')
    for pred in preds:
        b = pred.bbox
        x1 = b.minx * inv
        y1 = b.miny * inv
        w  = (b.maxx - b.minx) * inv
        h  = (b.maxy - b.miny) * inv
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=1.5,
            edgecolor=COLORS[conf],
            facecolor=COLORS[conf] + '22',
        )
        ax.add_patch(rect)

plt.suptitle(
    f'{img_name}  |  SAHI + YOLOv8m (COWC Potsdam fine-tune)  |  зона {area_m:.0f}м × {area_m:.0f}м',
    color='white', fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()

os.remove(TMP_PATCH)
print(f'\nГотово → {OUT_PATH}')
import subprocess
subprocess.Popen(['open', OUT_PATH])