"""
Road Detection — U-Net trained on OSM labels
Pipeline:
  1. Для каждого тайла получаем маску дорог из OSM (osmnx → rasterize)
  2. Нарезаем 512×512 патчи → датасет
  3. Обучаем U-Net (ResNet18 encoder, ImageNet init) 20 эпох
  4. Инференс на целевом тайле скользящим окном
  5. PNG визуализация + сохраняем модель
"""

import warnings; warnings.filterwarnings('ignore')
import os, json, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.warp import transform_bounds
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
import pyproj
import osmnx as ox
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────
#  НАСТРОЙКИ
# ─────────────────────────────────────────

INFER_TILE   = 'Dataset/Astana/Astana_4.tif'   # тайл для инференса
CROP_SIZE    = 4000    # пикс, центральный кроп из каждого тайла
PATCH_SIZE   = 512     # размер патча для обучения
STRIDE       = 256     # шаг при нарезке патчей
ROAD_BUFFER  = 6.0     # буфер вокруг центральной линии дороги, метры
EPOCHS       = 25
BATCH_SIZE   = 8
LR           = 3e-4
OUR_GSD      = 0.054   # м/пикс

TILES = [
    'Dataset/Astana/Astana_1.tif',
    'Dataset/Astana/Astana_2.tif',
    'Dataset/Astana/Astana_3.tif',
    'Dataset/Astana/Astana_4.tif',
    'Dataset/Astana/Astana_5.tif',
    'Dataset/Astana/Astana_6.tif',
    'Dataset/Astana/Astana_7.tif',
    'Dataset/Astana/Astana_8.tif',
    'Dataset/Astana/Astana_9.tif',
    'Dataset/Astana/Astana_ 10.tif',
]

# ─────────────────────────────────────────
#  ПУТИ
# ─────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'road_ml_data')
MODEL_PATH  = os.path.join(BASE_DIR, 'road_unet.pth')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

_ts    = datetime.now().strftime('%H%M%S')
_name  = os.path.splitext(os.path.basename(INFER_TILE))[0].strip()
OUT_PNG = os.path.join(RESULTS_DIR, f'roads_ml__{_name}__{_ts}.png')

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

ox.settings.log_console = False
ox.settings.use_cache   = True

# ─────────────────────────────────────────
#  ШАГ 1: Генерация масок из OSM
# ─────────────────────────────────────────

def get_crop_info(tif_path):
    with rasterio.open(os.path.join(BASE_DIR, tif_path)) as src:
        H, W   = src.height, src.width
        cx, cy = W // 2, H // 2
        crop   = min(CROP_SIZE, cx, cy, W - cx, H - cy)
        x0, y0 = cx - crop // 2, cy - crop // 2
        win    = rasterio.windows.Window(x0, y0, crop, crop)
        rgb    = np.moveaxis(src.read([1, 2, 3], window=win), 0, -1).astype(np.uint8)
        win_tf = src.window_transform(win)
        crs    = src.crs
        if crs.to_epsg() != 4326:
            bounds = transform_bounds(crs, 'EPSG:4326',
                                      win_tf.c, win_tf.f + win_tf.e * crop,
                                      win_tf.c + win_tf.a * crop, win_tf.f)
        else:
            bounds = (win_tf.c, win_tf.f + win_tf.e * crop,
                      win_tf.c + win_tf.a * crop, win_tf.f)
    return rgb, bounds, crop

def osm_road_mask(bounds_wgs, crop_px):
    lon_min, lat_min, lon_max, lat_max = bounds_wgs
    try:
        G = ox.graph_from_bbox(
            bbox=(lon_min, lat_min, lon_max, lat_max),
            network_type='all', retain_all=True,
        )
        edges = ox.graph_to_gdfs(G, nodes=False).to_crs('EPSG:32642')
    except Exception:
        return np.zeros((crop_px, crop_px), dtype=np.uint8)

    if len(edges) == 0:
        return np.zeros((crop_px, crop_px), dtype=np.uint8)

    # Буферизуем центральные линии → полигоны ширины дороги
    proj_fwd = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32642', always_xy=True).transform
    proj_inv = pyproj.Transformer.from_crs('EPSG:32642', 'EPSG:4326', always_xy=True).transform

    from shapely.geometry import box as shp_box
    # bounds в EPSG:32642
    ll = shp_transform(proj_fwd, shp_box(lon_min, lat_min, lon_max, lat_max))
    xmin, ymin, xmax, ymax = ll.bounds

    polys = []
    for geom in edges.geometry:
        buf = geom.buffer(ROAD_BUFFER)
        polys.append(buf)

    # Растеризуем в crop_px × crop_px
    tf = from_bounds(xmin, ymin, xmax, ymax, crop_px, crop_px)
    mask = rasterize(
        [(g, 1) for g in polys],
        out_shape=(crop_px, crop_px),
        transform=tf,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    # rasterize y-axis: top = ymax, flip if needed
    return mask

print('\n=== ШАГ 1: Генерация масок OSM ===')
pairs = []  # list of (rgb ndarray, mask ndarray)

for tif in TILES:
    tag = os.path.splitext(os.path.basename(tif))[0].strip()
    img_path  = os.path.join(DATA_DIR, f'{tag}_img.npy')
    mask_path = os.path.join(DATA_DIR, f'{tag}_mask.npy')

    if os.path.exists(img_path) and os.path.exists(mask_path):
        rgb  = np.load(img_path)
        mask = np.load(mask_path)
        road_px = mask.sum()
        print(f'  {tag}: загружено из кэша, road px={road_px}')
    else:
        try:
            rgb, bounds, crop = get_crop_info(tif)
        except Exception as e:
            print(f'  {tag}: пропускаю ({e})')
            continue
        print(f'  {tag}: запрашиваю OSM...', end=' ', flush=True)
        mask = osm_road_mask(bounds, crop)
        road_px = mask.sum()
        print(f'road px={road_px}')
        np.save(img_path,  rgb)
        np.save(mask_path, mask)

    if mask.sum() > 500:   # пропускаем тайлы без дорог
        pairs.append((rgb, mask))

print(f'\nТайлов с дорогами: {len(pairs)}')

# ─────────────────────────────────────────
#  ШАГ 2: Датасет из патчей
# ─────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class PatchDataset(Dataset):
    def __init__(self, pairs, patch, stride, augment=True):
        self.patches = []
        self.augment = augment
        for rgb, mask in pairs:
            H, W = rgb.shape[:2]
            for y in range(0, H - patch + 1, stride):
                for x in range(0, W - patch + 1, stride):
                    p_img  = rgb [y:y+patch, x:x+patch]
                    p_mask = mask[y:y+patch, x:x+patch]
                    if p_mask.sum() > 50:   # только патчи с дорогами
                        self.patches.append((p_img, p_mask))
        print(f'Патчей с дорогами: {len(self.patches)}')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, mask = self.patches[idx]
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = torch.from_numpy(img.copy()).permute(2, 0, 1)
        mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)
        if self.augment and torch.rand(1) > 0.5:
            img  = torch.flip(img,  [-1])
            mask = torch.flip(mask, [-1])
        if self.augment and torch.rand(1) > 0.5:
            img  = torch.flip(img,  [-2])
            mask = torch.flip(mask, [-2])
        return img, mask

print('\n=== ШАГ 2: Нарезка патчей ===')
dataset = PatchDataset(pairs, PATCH_SIZE, STRIDE)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ─────────────────────────────────────────
#  ШАГ 3: Обучение U-Net
# ─────────────────────────────────────────

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss  = smp.losses.SoftBCEWithLogitsLoss()

def loss_fn(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

print(f'\n=== ШАГ 3: Обучение U-Net ({EPOCHS} эпох, device={DEVICE}) ===')
if os.path.exists(MODEL_PATH):
    print(f'Модель найдена: {MODEL_PATH} — пропускаю обучение')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    if len(dataset) == 0:
        print('Нет патчей с дорогами — обучение невозможно')
        exit(1)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = loss_fn(pred, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        print(f'  Epoch {epoch:2d}/{EPOCHS}  loss={avg:.4f}')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Модель сохранена: {MODEL_PATH}')

# ─────────────────────────────────────────
#  ШАГ 4: Инференс скользящим окном
# ─────────────────────────────────────────

print(f'\n=== ШАГ 4: Инференс → {INFER_TILE} ===')
rgb_infer, bounds_infer, crop_infer = get_crop_info(INFER_TILE)
H, W = rgb_infer.shape[:2]

pred_sum   = np.zeros((H, W), dtype=np.float32)
pred_count = np.zeros((H, W), dtype=np.float32)

model.eval()
INF_STRIDE = PATCH_SIZE // 2

with torch.no_grad():
    ys = list(range(0, H - PATCH_SIZE + 1, INF_STRIDE)) + [H - PATCH_SIZE]
    xs = list(range(0, W - PATCH_SIZE + 1, INF_STRIDE)) + [W - PATCH_SIZE]
    for y in set(ys):
        for x in set(xs):
            patch = rgb_infer[y:y+PATCH_SIZE, x:x+PATCH_SIZE].astype(np.float32) / 255.0
            patch = (patch - MEAN) / STD
            t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            out = torch.sigmoid(model(t)).squeeze().cpu().numpy()
            pred_sum  [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += out
            pred_count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

road_prob = pred_sum / np.maximum(pred_count, 1)
road_mask = (road_prob > 0.4).astype(np.uint8)
print(f'Дорог найдено: {road_mask.sum()} px  ({road_mask.mean()*100:.1f}% площади)')

# ─────────────────────────────────────────
#  ШАГ 5: Визуализация
# ─────────────────────────────────────────

area_m = crop_infer * OUR_GSD
fig, axes = plt.subplots(1, 3, figsize=(33, 11))
fig.patch.set_facecolor('#0d0d0d')

axes[0].imshow(rgb_infer)
axes[0].set_title(f'Оригинал  {crop_infer}×{crop_infer}px = {area_m:.0f}м²',
                  color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(rgb_infer)
overlay = np.zeros((*road_mask.shape, 4), dtype=np.float32)
overlay[road_mask == 1] = [0, 0.9, 1.0, 0.55]
axes[1].imshow(overlay)
axes[1].set_title('U-Net Road Segmentation', color='#00e5ff', fontsize=12)
axes[1].axis('off')

axes[2].imshow(road_prob, cmap='hot', vmin=0, vmax=1)
axes[2].set_title('Confidence Map', color='#ffaa00', fontsize=12)
axes[2].axis('off')

plt.suptitle(f'{os.path.basename(INFER_TILE)}  |  Road U-Net (OSM labels, {EPOCHS} epochs)',
             color='white', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()

print(f'\nВизуализация → {OUT_PNG}')
subprocess.Popen(['open', OUT_PNG])
