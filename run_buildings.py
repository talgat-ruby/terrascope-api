"""
Building Detection — Microsoft Building Footprints (Kazakhstan)
Источник: Microsoft Global Building Footprints (free, ~1m accuracy)
Вход: любой GeoTIFF снимок в пределах Казахстана
Выход: PNG визуализация + GeoJSON полигоны зданий
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np, rasterio, os, json, gzip, csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection
import subprocess
from shapely.geometry import shape, box, mapping
import math

# ─────────────────────────────────────────
#  НАСТРОЙКИ — меняй здесь
# ─────────────────────────────────────────

IMAGE_PATH = 'Dataset/Astana/Astana_4.tif'   # любой тайл

# Размер зоны в пикселях (4000px ≈ 216м при 5.4см/пикс)
CROP_SIZE  = 4000

# Сдвиг от центра снимка в пикселях
OFFSET_X   = 0
OFFSET_Y   = 0

# Минимальная площадь здания в м² (фильтр шума)
MIN_AREA_M2 = 20

# ─────────────────────────────────────────
#  ПУТИ
# ─────────────────────────────────────────

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(BASE_DIR, 'results')
MSFT_DIR       = os.path.join(BASE_DIR, 'msft_buildings')
MSFT_LINKS_CSV = os.path.join(MSFT_DIR, 'dataset-links.csv')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MSFT_DIR, exist_ok=True)

_ts      = datetime.now().strftime('%H%M%S')
_name    = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
_suffix  = f'dx{OFFSET_X}_dy{OFFSET_Y}' if (OFFSET_X or OFFSET_Y) else 'center'
OUT_PNG  = os.path.join(RESULTS_DIR, f'buildings__{_name}__{_suffix}__{_ts}.png')
OUT_JSON = os.path.join(RESULTS_DIR, f'buildings__{_name}__{_suffix}__{_ts}.geojson')

OUR_GSD  = 0.054  # м/пикс

# ─────────────────────────────────────────
#  1. Читаем патч из снимка
# ─────────────────────────────────────────

print(f'\nСнимок: {IMAGE_PATH}')
with rasterio.open(os.path.join(BASE_DIR, IMAGE_PATH)) as src:
    H, W   = src.height, src.width
    cx     = W // 2 + OFFSET_X
    cy     = H // 2 + OFFSET_Y
    crop   = min(CROP_SIZE, cx, cy, W - cx, H - cy)
    x0, y0 = cx - crop // 2, cy - crop // 2
    win    = rasterio.windows.Window(x0, y0, crop, crop)
    rgb    = np.moveaxis(src.read([1, 2, 3], window=win), 0, -1).astype(np.uint8)
    win_tf = src.window_transform(win)
    crs    = src.crs

    # Bounds in WGS84
    from rasterio.warp import transform_bounds
    if crs.to_epsg() != 4326:
        bounds_wgs = transform_bounds(crs, 'EPSG:4326',
                                      win_tf.c, win_tf.f + win_tf.e * crop,
                                      win_tf.c + win_tf.a * crop, win_tf.f)
    else:
        bounds_wgs = (
            win_tf.c,
            win_tf.f + win_tf.e * crop,
            win_tf.c + win_tf.a * crop,
            win_tf.f,
        )

area_m = crop * OUR_GSD
print(f'Зона: {crop}×{crop} пикс = {area_m:.0f}м × {area_m:.0f}м')
print(f'WGS84 bounds: lon {bounds_wgs[0]:.5f}–{bounds_wgs[2]:.5f}, lat {bounds_wgs[1]:.5f}–{bounds_wgs[3]:.5f}')

# ─────────────────────────────────────────
#  2. Вычисляем нужные quadkeys
# ─────────────────────────────────────────

def lat_lon_to_quadkey(lat, lon, zoom):
    x = int((lon + 180) / 360 * (2 ** zoom))
    sin_lat = math.sin(math.radians(lat))
    y = int((0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (2 ** zoom))
    qk = ''
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask: digit += 1
        if y & mask: digit += 2
        qk += str(digit)
    return qk

def qk_to_bounds(qk):
    x = y = 0
    zoom = len(qk)
    for i, c in enumerate(qk):
        mask = 1 << (zoom - 1 - i)
        if c in ('1', '3'): x |= mask
        if c in ('2', '3'): y |= mask
    n = 2 ** zoom
    lon_min = x / n * 360 - 180
    lon_max = (x + 1) / n * 360 - 180
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max

roi = box(*bounds_wgs)

# Find all quadkeys from dataset-links.csv that overlap our ROI
def get_needed_quadkeys():
    import urllib.request
    links_url = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"

    if not os.path.exists(MSFT_LINKS_CSV):
        print('Скачиваю индекс Microsoft Building Footprints...')
        urllib.request.urlretrieve(links_url, MSFT_LINKS_CSV)

    needed = {}
    with open(MSFT_LINKS_CSV, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3 or parts[0] != 'Kazakhstan':
                continue
            qk = parts[1]
            b = qk_to_bounds(qk)
            if box(*b).intersects(roi):
                needed[qk] = parts[2]
    return needed

needed_qks = get_needed_quadkeys()
print(f'\nНужно quadkeys: {list(needed_qks.keys())}')

# ─────────────────────────────────────────
#  3. Скачиваем и фильтруем здания
# ─────────────────────────────────────────

import urllib.request

def load_buildings_from_file(gz_path):
    buildings = []
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            full = ','.join(row).replace('geometry:', '"geometry":').replace('coordinates:', '"coordinates":')
            try:
                feat = json.loads(full)
                poly = shape(feat['geometry'])
                if poly.intersects(roi):
                    buildings.append((poly, feat.get('properties', {})))
            except:
                pass
    return buildings

all_buildings = []
for qk, url in needed_qks.items():
    local_path = os.path.join(MSFT_DIR, f'{qk}.csv.gz')
    if not os.path.exists(local_path):
        print(f'Скачиваю quadkey {qk}...')
        urllib.request.urlretrieve(url, local_path)
        print(f'  Готово: {os.path.getsize(local_path) // 1024 // 1024}MB')
    else:
        print(f'Quadkey {qk} уже скачан ({os.path.getsize(local_path) // 1024 // 1024}MB)')

    print(f'Фильтрую здания в пределах снимка...')
    blds = load_buildings_from_file(local_path)
    all_buildings.extend(blds)
    print(f'  Найдено в этом тайле: {len(blds)}')

print(f'\nВсего зданий в зоне снимка: {len(all_buildings)}')

# ─────────────────────────────────────────
#  4. Фильтрация по площади → GeoJSON
# ─────────────────────────────────────────

features = []
for poly, props in all_buildings:
    clipped = poly.intersection(roi)
    if clipped.is_empty:
        continue

    # Площадь в м²
    if crs and crs.to_epsg() == 4326:
        # WGS84: грубая оценка через длину 1° в м
        lat_m = 111_320.0
        lon_m = 111_320.0 * math.cos(math.radians(poly.centroid.y))
        area_m2 = poly.area * lat_m * lon_m
    else:
        area_m2 = poly.area

    if area_m2 < MIN_AREA_M2:
        continue

    features.append({
        'type': 'Feature',
        'geometry': mapping(clipped),
        'properties': {
            'class': 'building',
            'area_m2': round(area_m2, 1),
            'source': 'Microsoft Global Building Footprints',
        }
    })

geojson = {'type': 'FeatureCollection', 'features': features}
with open(OUT_JSON, 'w') as f:
    json.dump(geojson, f)

print(f'Зданий после фильтрации (площадь ≥ {MIN_AREA_M2}м²): {len(features)}')

# ─────────────────────────────────────────
#  5. Визуализация
# ─────────────────────────────────────────

# Pixel coordinates for polygon overlay
lon0 = bounds_wgs[0]
lat0 = bounds_wgs[3]   # top-left lat (north)
lon1 = bounds_wgs[2]
lat1 = bounds_wgs[1]   # bottom-right lat (south)

def geo_to_px(lon, lat):
    px = (lon - lon0) / (lon1 - lon0) * crop
    py = (lat - lat0) / (lat1 - lat0) * crop
    return px, py

fig, axes = plt.subplots(1, 2, figsize=(22, 11))
fig.patch.set_facecolor('#0d0d0d')

# Left: original image
axes[0].imshow(rgb)
axes[0].set_title(f'Оригинал\n{crop}×{crop} пикс = {area_m:.0f}м × {area_m:.0f}м',
                  color='white', fontsize=12)
axes[0].axis('off')

# Right: buildings overlay
axes[1].imshow(rgb)
axes[1].set_title(f'Microsoft Building Footprints\nНайдено зданий: {len(features)} (площадь ≥ {MIN_AREA_M2}м²)',
                  color='#44aaff', fontsize=12)
axes[1].axis('off')

polys_mpl = []
for feat in features:
    geom = shape(feat['geometry'])
    if geom.geom_type == 'Polygon':
        geoms = [geom]
    else:
        geoms = list(geom.geoms)

    for g in geoms:
        coords = list(g.exterior.coords)
        px_coords = [geo_to_px(lon, lat) for lon, lat in coords]
        p = MplPoly(px_coords, closed=True)
        polys_mpl.append(p)

if polys_mpl:
    col = PatchCollection(polys_mpl,
                          facecolor=(0.27, 0.67, 1.0, 0.4),
                          edgecolor=(0.2, 0.7, 1.0, 0.9),
                          linewidth=0.8)
    axes[1].add_collection(col)

plt.suptitle(
    f'{os.path.basename(IMAGE_PATH)}  |  Microsoft Building Footprints  |  buildings',
    color='white', fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()

print(f'\nВизуализация → {OUT_PNG}')
print(f'GeoJSON      → {OUT_JSON}')
subprocess.Popen(['open', OUT_PNG])
