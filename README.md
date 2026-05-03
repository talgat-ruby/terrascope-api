# Car Detection on Satellite Imagery — NURIS Hackathon

Детекция автомобилей на аэроснимках Алматы и Астаны (GSD ≈ 5.4 см/пикс).

**Модель**: YOLOv8m fine-tuned на COWC Potsdam (15 см/пикс)  
**Метод инференса**: SAHI (Slicing Aided Hyper Inference) — автоматический тайлинг больших снимков  
**Результат**: mAP50 = **0.939** на валидационном сете COWC

---

## Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Создать виртуальное окружение и установить зависимости
```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

> ⚠️ Установка torch может занять 5-10 минут

### 3. Положить снимки
Создай папку `Dataset/` и положи туда GeoTIFF файлы:
```
Dataset/
  Almaty/
    Almaty_1.tif
    Almaty_2.tif
    ...
  Astana/
    Astana_1.tif
    ...
```

### 4. Запустить детекцию
```bash
python run_detection.py
```

Результат автоматически откроется в Preview и сохранится в папке `results/`.

---

## Настройки в `run_detection.py`

```python
IMAGE_PATH = 'Dataset/Almaty/Almaty_1.tif'  # какой снимок

CROP_SIZE = 5555   # размер зоны (5555 пикс ≈ 300м × 300м)
OFFSET_X  = 0      # сдвиг центра вправо (пикс)
OFFSET_Y  = 0      # сдвиг центра вниз (пикс)

THRESHOLDS = [0.25, 0.15, 0.10]  # три порога confidence для сравнения
```

---

## Как это работает

```
GeoTIFF (5.4 см/пикс)
        ↓
  Вырезаем патч (300м × 300м)
        ↓
  Downsample ×0.36 → эквивалент 15 см/пикс (GSD обучения)
        ↓
  SAHI sliding window (640×640, overlap 20%)
        ↓
  YOLOv8m inference + NMS мерж на стыках
        ↓
  Визуализация трёх порогов рядом → results/*.png
```

---

## Модель

- Файл: `best.pt` (50 MB, включён в репозиторий)
- Архитектура: YOLOv8m (Medium)
- Обучен на: COWC Potsdam subset (~2825 машин, 187 тайлов 640×640)
- Эпох: 30
- mAP50: 0.939 | Precision: 0.940 | Recall: 0.882

---

## Известные ограничения

- Модель обучена на легковых авто — **автобусы и грузовики пропускаются**
- Domain gap: Потсдам (Германия) vs Алматы/Астана — часть машин пропускается
- Машины под углом 45° детектируются хуже (нет OBB в обучении)
- Для лучших результатов нужна аннотация своих снимков + дообучение

---

## Детекция дорог (U-Net + OSM)

### Запуск

```bash
python run_roads_ml.py
```

Результат сохраняется в `results/roads_ml__<tile>__<time>.png` и автоматически открывается.

### Что делает скрипт

```
Dataset/Astana/*.tif  (10 тайлов)
        ↓
  Запрашиваем дороги из OSM (osmnx) → растеризуем маску дорог
  Кэш numpy-патчей сохраняется в road_ml_data/ (пропускается при повторном запуске)
        ↓
  Нарезаем 512×512 патчи → обучаем U-Net (ResNet34 encoder, ImageNet init)
  Обученная модель: road_unet.pth (включена в репозиторий — обучение пропускается)
        ↓
  Инференс скользящим окном на тайле Astana_4.tif
        ↓
  PNG: оригинал | сегментация | confidence map
```

### Требования к данным

Файлы тайлов должны лежать в:
```
Dataset/Astana/Astana_1.tif  ...  Astana_ 10.tif
```
Снимки **не включены в репозиторий** (слишком большой размер) — нужно добавить вручную.

### Настройки в `run_roads_ml.py`

```python
INFER_TILE  = 'Dataset/Astana/Astana_4.tif'  # тайл для инференса
CROP_SIZE   = 4000    # центральный кроп (пикс)
ROAD_BUFFER = 6.0     # ширина дороги в метрах
EPOCHS      = 25      # эпох обучения (игнорируется если road_unet.pth уже есть)
```

### Модель

- Файл: `road_unet.pth` (93 МБ, включён в репозиторий)
- Архитектура: U-Net + ResNet34 encoder (ImageNet init)
- Метки: OSM road network, растеризованный буфер 6 м
- Эпох: 25 | Устройство: MPS/CUDA/CPU (автоопределение)
