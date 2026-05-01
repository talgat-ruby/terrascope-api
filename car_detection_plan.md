# Car Detection — Progress Tracker

**Цель**: детектировать все транспортные средства как один класс `car` на аэроснимках Алматы/Астаны  
**GSD данных**: 5.4 см/пиксель | Размер тайла: 15 377 × 11 259 пикс | Площадь: ~0.5 км²  
**Автомобиль в пикселях**: седан ~4.5м → **~83 пикс** в длину — отлично виден  

---

## Выбор стратегии

### Модель: YOLOv8m-OBB (Oriented Bounding Box)
- Pretrained на **DOTA v1** (классы `small-vehicle` + `large-vehicle` → мержим в `car`)
- OBB = повёрнутые bbox — критично для аэросъёмки (машины стоят под разными углами)
- Официальные веса от Ultralytics, MIT License
- Альтернатива: YOLOv8l-OBB если мощности позволяют

### Dataset: COWC (Cars Overhead with Context)
- GSD: **15 см/пикс** — ближайший публично доступный к вашим 5.4 см
- ~33 000 размеченных машин, 6 городов (Columbus, Potsdam, Selwyn, Toronto, Utah, Vancouver)
- Одиночный класс — идеально под нашу задачу
- Лицензия: CC BY-NC-SA
- Для файн-тюна: масштаб изображений с 15 см → 5.4 см (resize ×2.8)

### Почему не брать DOTA напрямую?
DOTA снят с GSD 0.3–1.0 м — машины там 5–15 пикселей. У нас 83 пикселя.
Pretrained веса на DOTA используем как **стартовую точку**, дообучаем на COWC.

---

## Фазы работы

### Фаза 0 — Окружение
- [ ] 0.1 Установить ultralytics в venv
- [ ] 0.2 Проверить torch + GPU
- [ ] 0.3 Проверить запуск `yolo predict` на тестовой картинке

```bash
# Активируем venv
source Dataset/.venv_nuris/bin/activate

# Устанавливаем ultralytics
pip install ultralytics supervision

# Проверка
python -c "from ultralytics import YOLO; print('OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
```

---

### Фаза 1 — Скачать модель и проверить zero-shot

**Цель**: запустить pretrained YOLOv8m-OBB на одном патче снимка и посмотреть что видит модель без всякого обучения.

- [ ] 1.1 Скачать веса YOLOv8m-OBB (pretrained DOTA)
- [ ] 1.2 Вырезать тестовый патч из Almaty_1.tif (640×640 пикс)
- [ ] 1.3 Запустить inference на патче
- [ ] 1.4 Визуализировать результат — сохранить картинку

```python
from ultralytics import YOLO
import rasterio, numpy as np
from PIL import Image

# Модель скачается автоматически (~52 MB)
model = YOLO("yolov8m-obb.pt")

# Вырезаем патч из большого снимка
with rasterio.open("Dataset/Almaty/Almaty_1.tif") as src:
    # читаем первые 3 канала (RGB), патч 640x640 из центра
    h, w = src.height, src.width
    window = rasterio.windows.Window(w//2, h//2, 640, 640)
    rgb = src.read([1, 2, 3], window=window)  # (3, 640, 640)
    rgb = np.moveaxis(rgb, 0, -1)             # (640, 640, 3)

img = Image.fromarray(rgb.astype(np.uint8))
img.save("test_patch.png")

# Inference
results = model.predict("test_patch.png", conf=0.25, iou=0.45)
results[0].save("test_patch_result.png")
print(f"Найдено объектов: {len(results[0].obb.cls)}")
```

**Ожидаемый результат**: модель увидит машины, но возможны пропуски — DOTA GSD сильно отличается.  
**Критерий прохождения**: хотя бы ~30% реальных машин детектируются визуально.

---

### Фаза 2 — Скачать и подготовить датасет COWC

**Цель**: получить размеченный датасет с GSD близким к нашему.

- [ ] 2.1 Скачать COWC Potsdam subset (самый качественный, 15 см/пикс)
- [ ] 2.2 Конвертировать аннотации в YOLO-OBB формат (один класс `car`)
- [ ] 2.3 Сделать train/val split (80/20)
- [ ] 2.4 Просмотреть 10 случайных изображений с боксами — убедиться что разметка корректна
- [ ] 2.5 Посчитать базовую статистику: кол-во машин, средний размер bbox

**Скачать COWC:**
```bash
# Potsdam subset (~2 GB)
wget https://gdo152.llnl.gov/cowc/download/datasets/Potsdam_ISPRS.tbz
tar -xjf Potsdam_ISPRS.tbz

# Или Toronto subset (~1.5 GB)  
wget https://gdo152.llnl.gov/cowc/download/datasets/Toronto_ISPRS.tbz
```

**Конвертация аннотаций в YOLO-OBB формат:**
```python
# COWC хранит аннотации как центры (x, y) — нужно добавить размер bbox
# Средний размер машины в COWC Potsdam (15 см/пикс): ~20×10 пикс (3м × 1.5м)
# YOLO OBB формат: class cx cy w h angle (normalized)

import os, glob
import numpy as np
from PIL import Image

COWC_IMG_DIR = "Potsdam_ISPRS/Images/"
COWC_ANN_DIR = "Potsdam_ISPRS/Annotations/"  # .txt с центрами (x y per line)
OUT_IMG_DIR = "cowc_yolo/images/"
OUT_LBL_DIR = "cowc_yolo/labels/"
CAR_SIZE_PX = (20, 10)  # ширина, высота в пикселях при 15 см/GSD

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

for ann_path in glob.glob(f"{COWC_ANN_DIR}/*.txt"):
    img_name = os.path.basename(ann_path).replace(".txt", ".png")
    img_path = f"{COWC_IMG_DIR}/{img_name}"
    if not os.path.exists(img_path):
        continue
    
    img = Image.open(img_path)
    W, H = img.size
    
    centers = np.loadtxt(ann_path).reshape(-1, 2)  # (N, 2)
    lines = []
    for cx, cy in centers:
        # нормализуем
        cx_n, cy_n = cx / W, cy / H
        w_n, h_n = CAR_SIZE_PX[0] / W, CAR_SIZE_PX[1] / H
        # YOLO-OBB: class cx cy w h angle
        lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f} 0.0")
    
    out_lbl = f"{OUT_LBL_DIR}/{img_name.replace('.png', '.txt')}"
    with open(out_lbl, "w") as f:
        f.write("\n".join(lines))
    img.save(f"{OUT_IMG_DIR}/{img_name}")

print("Конвертация завершена")
```

**Посмотреть несколько изображений:**
```python
import matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image
import glob, random, numpy as np

imgs = random.sample(glob.glob("cowc_yolo/images/*.png"), 10)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for ax, img_path in zip(axes.flatten(), imgs):
    img = np.array(Image.open(img_path))
    H, W = img.shape[:2]
    lbl_path = img_path.replace("images", "labels").replace(".png", ".txt")
    
    ax.imshow(img)
    ax.set_title(os.path.basename(img_path)[:20], fontsize=7)
    ax.axis("off")
    
    if os.path.exists(lbl_path):
        for line in open(lbl_path):
            parts = list(map(float, line.split()))
            cx, cy, w, h = parts[1]*W, parts[2]*H, parts[3]*W, parts[4]*H
            rect = patches.Rectangle((cx-w/2, cy-h/2), w, h,
                                       linewidth=1, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

plt.tight_layout()
plt.savefig("cowc_sample_visualization.png", dpi=150)
plt.show()
print("Сохранено: cowc_sample_visualization.png")
```

**Статистика датасета:**
```python
import glob, numpy as np

all_counts = []
all_sizes = []

for lbl in glob.glob("cowc_yolo/labels/*.txt"):
    data = open(lbl).readlines()
    all_counts.append(len(data))
    for line in data:
        parts = list(map(float, line.split()))
        all_sizes.append((parts[3], parts[4]))

print(f"Изображений: {len(all_counts)}")
print(f"Всего машин: {sum(all_counts)}")
print(f"Среднее на изображение: {np.mean(all_counts):.1f}")
print(f"Макс на изображение: {max(all_counts)}")
```

---

### Фаза 3 — Fine-tune модели на COWC

**Цель**: адаптировать YOLOv8m-OBB под наш тип данных (высокое разрешение, вид сверху, одна категория).

- [ ] 3.1 Создать dataset.yaml конфиг
- [ ] 3.2 Запустить fine-tune (50 эпох)
- [ ] 3.3 Наблюдать за loss в реальном времени
- [ ] 3.4 Выбрать best.pt checkpoint

```yaml
# cowc.yaml
path: cowc_yolo
train: images/train
val: images/val

nc: 1
names:
  0: car
```

```python
from ultralytics import YOLO

model = YOLO("yolov8m-obb.pt")  # стартуем с DOTA pretrained

results = model.train(
    data="cowc.yaml",
    epochs=50,
    imgsz=640,
    batch=16,          # уменьши до 8 если OOM
    lr0=1e-4,          # низкий LR для fine-tune
    warmup_epochs=3,
    patience=15,       # early stopping
    device="mps",      # Mac Apple Silicon; "cuda" для NVIDIA; "cpu" крайний случай
    project="runs/car_detection",
    name="cowc_finetune",
    save_period=10,    # сохранять каждые 10 эпох
    plots=True,        # графики loss/metrics автоматически
)
```

**Мониторинг во время обучения** — в другом терминале:
```bash
# Открыть TensorBoard
tensorboard --logdir runs/car_detection
# Открыть в браузере: http://localhost:6006
```

---

### Фаза 4 — Inference на полном снимке Алматы (SAHI)

**Цель**: запустить модель на реальных данных с правильным тайлингом через SAHI.

Проблема: снимок 15 377×11 259 пикс — в YOLO не войдёт целиком.  
Решение: **SAHI** (Slicing Aided Hyper Inference) — автоматический sliding window + NMS мерж на границах.

**Почему SAHI лучше ручного тайлинга:**
- Встроенный постпроцессинг дублей на стыках (NMM / Greedy NMS)
- Не нужно вручную писать логику overlap и мержа
- Прямая интеграция с YOLOv8
- Даёт +15-30% Recall на мелких объектах

- [ ] 4.1 Экспортировать снимок в PNG (или читать напрямую через rasterio)
- [ ] 4.2 Запустить SAHI inference с tile_size=640, overlap_ratio=0.2
- [ ] 4.3 Конвертировать pixel coords → GPS координаты (lat/lon)
- [ ] 4.4 Экспортировать в GeoJSON
- [ ] 4.5 Открыть в QGIS — визуальная проверка

```python
import rasterio, numpy as np, json
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Загружаем модель через SAHI
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="runs/car_detection/cowc_finetune/weights/best.pt",
    confidence_threshold=0.3,
    device="mps",  # или "cuda" / "cpu"
)

# Читаем снимок и сохраняем как PNG для SAHI
with rasterio.open("Dataset/Almaty/Almaty_1.tif") as src:
    transform = src.transform
    rgb = src.read([1, 2, 3])
    rgb = np.moveaxis(rgb, 0, -1).astype(np.uint8)

Image.fromarray(rgb).save("almaty1_rgb.png")

# SAHI inference — автоматический тайлинг + NMS мерж
result = get_sliced_prediction(
    "almaty1_rgb.png",
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Конвертируем pixel bbox → GPS + экспорт GeoJSON
features = []
for pred in result.object_prediction_list:
    bbox = pred.bbox  # (minx, miny, maxx, maxy) в пикселях
    cx_px = (bbox.minx + bbox.maxx) / 2
    cy_px = (bbox.miny + bbox.maxy) / 2
    lon, lat = rasterio.transform.xy(transform, cy_px, cx_px)
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "class": "car",
            "confidence": round(pred.score.value * 100, 1),
            "source": "Almaty_1.tif"
        }
    })

geojson = {"type": "FeatureCollection", "features": features}
with open("almaty_cars.geojson", "w") as f:
    json.dump(geojson, f)

print(f"Найдено машин: {len(features)}")
print("Сохранено: almaty_cars.geojson")
```

---

### Фаза 5 — Измерить прогресс

**Цель**: понять насколько хорошо работает модель.

- [ ] 5.1 Ручная проверка на 3 тайлах — считаем Precision/Recall вручную
- [ ] 5.2 Запустить model.val() на val-сете COWC — получить mAP
- [ ] 5.3 Записать результаты в таблицу ниже
- [ ] 5.4 Сравнить: zero-shot (DOTA only) vs fine-tune (COWC)

**Метрики:**

```python
# Автоматические метрики на COWC val set
from ultralytics import YOLO

model = YOLO("runs/car_detection/cowc_finetune/weights/best.pt")
metrics = model.val(data="cowc.yaml")

print(f"mAP@50:    {metrics.box.map50:.3f}")
print(f"mAP@50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.p[0]:.3f}")
print(f"Recall:    {metrics.box.r[0]:.3f}")
```

**Таблица результатов** (заполнять по мере работы):

| Этап | Модель | Датасет | mAP@50 | Precision | Recall | Примечание |
|---|---|---|---|---|---|---|
| Zero-shot | YOLOv8m-OBB | DOTA pretrained | — | — | — | Визуальная оценка |
| Fine-tune v1 | YOLOv8m-OBB | COWC Potsdam | — | — | — | 50 эпох |
| Fine-tune v2 | YOLOv8l-OBB | COWC Potsdam | — | — | — | Larger model |

**Ручная проверка** (заполнить для 3 тайлов Алматы):

| Тайл | Реальных машин (ручной счёт) | Детектировано | TP | FP | FN |
|---|---|---|---|---|---|
| Almaty_1 patch A | — | — | — | — | — |
| Almaty_1 patch B | — | — | — | — | — |
| Almaty_1 patch C | — | — | — | — | — |

---

## Целевые метрики

| Метрика | Приемлемо | Хорошо | Отлично |
|---|---|---|---|
| mAP@50 (COWC val) | > 0.50 | > 0.70 | > 0.85 |
| Precision | > 0.60 | > 0.75 | > 0.85 |
| Recall | > 0.55 | > 0.70 | > 0.80 |
| Визуальная точность (Almaty) | > 40% | > 60% | > 75% |

---

## Текущий статус

```
Фаза 0 — Окружение       [x] DONE — ultralytics 8.4.45, sahi 0.11.36, MPS ✓
Фаза 1 — Zero-shot тест  [x] DONE — результаты ниже
Фаза 2 — COWC датасет    [x] DONE — 187 тайлов, 2825 машин, cowc_samples.png
Фаза 3 — Fine-tune        [~] ИДЁТ — YOLOv8m, 50 эпох, MPS, runs/car_detection/cowc_finetune/
Фаза 4 — Inference Алматы [ ] не начато
Фаза 5 — Метрики          [ ] не начато
```

### Результаты Фазы 1 (zero-shot, Almaty_1.tif)

| Конфигурация | Патч | Найдено машин |
|---|---|---|
| Оригинальный GSD 5.4 см, патч 1280px | center | 1 |
| Симуляция DOTA GSD (downsample), покрытие 192m×192m | center | 4 |
| Симуляция DOTA GSD (downsample), покрытие 192m×192m | upper-left | 0 |
| Симуляция DOTA GSD (downsample), покрытие 192m×192m | upper-right | 1 |

**Ключевой вывод — GSD mismatch:**
- Модель DOTA обучена при GSD 0.3 м → машина = ~15 пикселей
- Наши снимки GSD = 5.4 см → машина = **83 пикселя**
- Zero-shot практически не работает — модель "не узнаёт" машины такого размера
- Визуализация: `zero_shot_analysis.png`

**Вывод**: необходим fine-tune на датасете с близким GSD → переходим к Фазе 2 (COWC)

---

## Полезные ссылки

| Ресурс | Ссылка |
|---|---|
| YOLOv8 OBB docs | https://docs.ultralytics.com/tasks/obb/ |
| COWC Dataset | https://gdo152.llnl.gov/cowc/ |
| DOTA Dataset | https://captain-whu.github.io/DOTA/ |
| DOTA pretrained weights | https://github.com/ultralytics/ultralytics (автозагрузка) |
| Roboflow COWC ready | https://universe.roboflow.com/search?q=COWC |
| keremberke HF model | https://huggingface.co/keremberke/yolov8m-satellite-vehicle-detection |

---

## Заметки

_Добавляй сюда наблюдения по мере работы_

- 
- 
