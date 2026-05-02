"""
Road Detection — OpenStreetMap via osmnx
Источник: OpenStreetMap (бесплатно, обновляется волонтёрами)
Вход:  любой GeoTIFF снимок в пределах Казахстана
Выход: PNG визуализация + GeoJSON дорог с типами
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np, rasterio, os, json, subprocess
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.warp import transform_bounds
from shapely.geometry import mapping
import math

# ─────────────────────────────────────────
#  НАСТРОЙКИ — меняй здесь
# ─────────────────────────────────────────

IMAGE_PATH = 'Dataset/Astana/Astana_2.tif'   # любой тайл

# Размер зоны в пикселях (4000px ≈ 216м при 5.4см/пикс)
CROP_SIZE  = 4000

# Сдвиг от центра снимка в пикселях
OFFSET_X   = 0
OFFSET_Y   = 0

# Цвет и толщина единственного класса "road"
ROAD_COLOR = '#00e5ff'   # яркий голубой — хорошо видно на любом фоне
ROAD_LW    = 2.5         # толщина линии

# ─────────────────────────────────────────
#  ПУТИ
# ─────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

_ts     = datetime.now().strftime('%H%M%S')
_name   = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
_suffix = f'dx{OFFSET_X}_dy{OFFSET_Y}' if (OFFSET_X or OFFSET_Y) else 'center'
OUT_PNG  = os.path.join(RESULTS_DIR, f'roads__{_name}__{_suffix}__{_ts}.png')
OUT_JSON = os.path.join(RESULTS_DIR, f'roads__{_name}__{_suffix}__{_ts}.geojson')

OUR_GSD = 0.054  # м/пикс

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
#  2. Скачиваем дороги из OSM
# ─────────────────────────────────────────

try:
    import osmnx as ox
except ImportError:
    print('\nУстановка osmnx...')
    import subprocess as sp
    sp.run(['pip', 'install', 'osmnx', '-q'], check=True)
    import osmnx as ox

ox.settings.log_console = False
ox.settings.use_cache   = True   # кэширует запросы локально

lon_min, lat_min, lon_max, lat_max = bounds_wgs
print('\nЗапрашиваю дороги из OpenStreetMap...')

try:
    # osmnx 2.x: bbox = (west, south, east, north)
    G = ox.graph_from_bbox(
        bbox=(lon_min, lat_min, lon_max, lat_max),
        network_type='all',
        retain_all=True,
    )
    edges = ox.graph_to_gdfs(G, nodes=False)
    print(f'Найдено сегментов дорог: {len(edges)}')
except Exception as e:
    print(f'Ошибка запроса OSM: {e}')
    edges = None

# ─────────────────────────────────────────
#  3. Сохраняем GeoJSON
# ─────────────────────────────────────────

features = []
if edges is not None and len(edges) > 0:
    edges_wgs = edges.to_crs('EPSG:4326')
    for _, row in edges_wgs.iterrows():
        geom = row.geometry
        hw   = row.get('highway', 'unclassified')
        if isinstance(hw, list):
            hw = hw[0]
        name = row.get('name', '')
        if isinstance(name, list):
            name = name[0] if name else ''
        features.append({
            'type': 'Feature',
            'geometry': mapping(geom),
            'properties': {
                'class': 'road',
                'name': name or '',
                'source': 'OpenStreetMap',
            }
        })

geojson = {'type': 'FeatureCollection', 'features': features}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(geojson, f, ensure_ascii=False)

# ─────────────────────────────────────────
#  4. Визуализация
# ─────────────────────────────────────────

def geo_to_px(lon, lat):
    lon0, lat_s, lon1, lat_n = bounds_wgs
    px = (lon - lon0) / (lon1 - lon0) * crop
    py = (lat - lat_n) / (lat_s - lat_n) * crop
    return px, py

fig, axes = plt.subplots(1, 2, figsize=(22, 11))
fig.patch.set_facecolor('#0d0d0d')

axes[0].imshow(rgb)
axes[0].set_title(f'Оригинал\n{crop}×{crop} пикс = {area_m:.0f}м × {area_m:.0f}м',
                  color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(rgb)
axes[1].set_title(f'OpenStreetMap Roads\nНайдено сегментов: {len(features)}',
                  color='#44ff88', fontsize=12)
axes[1].axis('off')

for feat in features:
    geom = feat['geometry']
    coords = geom.get('coordinates', [])
    if geom['type'] == 'LineString':
        coords_list = [coords]
    elif geom['type'] == 'MultiLineString':
        coords_list = coords
    else:
        continue
    for line in coords_list:
        xs = [geo_to_px(c[0], c[1])[0] for c in line]
        ys = [geo_to_px(c[0], c[1])[1] for c in line]
        axes[1].plot(xs, ys, color=ROAD_COLOR, linewidth=ROAD_LW, solid_capstyle='round')

from matplotlib.lines import Line2D
axes[1].legend(
    handles=[Line2D([0], [0], color=ROAD_COLOR, linewidth=2.5, label='road')],
    loc='lower right', framealpha=0.7, facecolor='#1a1a1a', labelcolor='white', fontsize=9,
)

plt.suptitle(
    f'{os.path.basename(IMAGE_PATH)}  |  OpenStreetMap Roads',
    color='white', fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()

print(f'\nВизуализация → {OUT_PNG}')
print(f'GeoJSON      → {OUT_JSON}')
subprocess.Popen(['open', OUT_PNG])
