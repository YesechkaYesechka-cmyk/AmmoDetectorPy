import os
import glob
import json
import argparse
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
import matplotlib.pyplot as plt

# ────────────────────────────── НАСТРОЙКИ ──────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Модель и метаданные будем класть в одну папку
MODEL_DIR = 'saved_ammo_model'            # директория для артефактов
MODEL_PATH = os.path.join(MODEL_DIR, 'model.keras')
CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.json')

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

# ───────────── выбор предобученной базы и препроцессинга ───────────────
def get_preprocess_and_base(base_model_name: str):
    name = base_model_name.lower()
    if name == 'vgg16':
        preprocess = vgg16_preprocess
        base_ctor = lambda: VGG16(weights='imagenet', include_top=False,
                                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        preprocess_name = 'vgg16'
    elif name == 'resnet50':
        preprocess = resnet50_preprocess
        base_ctor = lambda: ResNet50(weights='imagenet', include_top=False,
                                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        preprocess_name = 'resnet50'
    elif name == 'efficientnet':
        preprocess = effnet_preprocess
        base_ctor = lambda: EfficientNetB0(weights='imagenet', include_top=False,
                                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        preprocess_name = 'efficientnet'
    else:
        raise ValueError('Неизвестная базовая модель: используйте vgg16, resnet50 или efficientnet')

    def resize_with_pad_and_preprocess(x):
        x = tf.cast(x, tf.float32)
        x = tf.image.resize_with_pad(x, IMG_HEIGHT, IMG_WIDTH)
        x = preprocess(x)
        return x

    return resize_with_pad_and_preprocess, base_ctor, preprocess_name

# ───────────────────── авто-обнаружение классов по папкам ──────────────
def list_classes_from_dir(root_dir: str):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f'Каталог {root_dir} не найден')
    names = [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    if not names:
        raise RuntimeError(f'В {root_dir} нет подпапок классов')
    names = sorted(names)
    return names

def ensure_same_class_sets(train_classes, other_root, other_name):
    if not os.path.isdir(other_root):
        print(f'Внимание: {other_name} каталог {other_root} отсутствует — пропускаю проверку классов')
        return
    other = [d for d in os.listdir(other_root)
             if os.path.isdir(os.path.join(other_root, d)) and not d.startswith('.')]
    missing = set(train_classes) - set(other)
    extra = set(other) - set(train_classes)
    if missing:
        print(f'Предупреждение: в {other_name} отсутствуют классы: {sorted(missing)}')
    if extra:
        print(f'Предупреждение: в {other_name} найдены лишние классы (игнорируются): {sorted(extra)}')

# ───────────── чтение датасета через tf.data (мультикласс) ──────────────
IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')

def collect_files_and_labels(root_dir: str, class_names):
    filepaths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        cls_files = []
        for ext in IMAGE_EXTS:
            cls_files += glob.glob(os.path.join(cls_dir, ext))
            cls_files += glob.glob(os.path.join(cls_dir, '**', ext), recursive=True)
        filepaths.extend(cls_files)
        labels.extend([idx] * len(cls_files))
    return filepaths, labels

def make_dataset(root_dir: str, class_names, preprocess_fn, shuffle: bool, augment: bool):
    filepaths, labels = collect_files_and_labels(root_dir, class_names)
    if len(filepaths) == 0:
        raise FileNotFoundError(f'В {root_dir} не найдено изображений в папках {class_names}')

    paths_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))

    def load_image(path):
        img_bin = tf.io.read_file(path)
        img = tf.io.decode_image(img_bin, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return img

    aug_layers = keras.Sequential([
        layers.RandomFlip('horizontal', seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
        layers.RandomTranslation(0.1, 0.1, seed=SEED),
        layers.RandomContrast(0.1, seed=SEED),
    ])

    def map_fn(path, label):
        img = load_image(path)
        if augment:
            img = aug_layers(img, training=True)
        img = preprocess_fn(img)
        return img, label

    ds = tf.data.Dataset.zip((paths_ds, labels_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2048, len(filepaths)), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def create_datasets(base_model_name='vgg16', class_names=None):
    if class_names is None:
        class_names = list_classes_from_dir(TRAIN_DIR)
    ensure_same_class_sets(class_names, VAL_DIR, 'validation')
    ensure_same_class_sets(class_names, TEST_DIR, 'test')

    preprocess_fn, _, _ = get_preprocess_and_base(base_model_name)
    train_ds = make_dataset(TRAIN_DIR, class_names, preprocess_fn, shuffle=True,  augment=True)
    val_ds   = make_dataset(VAL_DIR,   class_names, preprocess_fn, shuffle=False, augment=False)
    test_ds  = make_dataset(TEST_DIR,  class_names, preprocess_fn, shuffle=False, augment=False)
    return train_ds, val_ds, test_ds, class_names

# ───────────────────────────── проверка данных ───────────────────────────
def check_dataset_integrity(class_names):
    print('Проверка целостности данных...')
    for cls in class_names:
        counts = {}
        for split, root in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
            dir_path = os.path.join(root, cls)
            cnt = 0
            if os.path.isdir(dir_path):
                for ext in IMAGE_EXTS:
                    cnt += len(glob.glob(os.path.join(dir_path, ext)))
                    cnt += len(glob.glob(os.path.join(dir_path, '**', ext), recursive=True))
            counts[split] = cnt
        print(f'{cls}: train={counts["train"]}, val={counts["val"]}, test={counts["test"]}')

# ───────────────────────────── построение модели ────────────────────────
def build_head(base_model, num_classes):
    return keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

def create_model(base_model_name='vgg16', num_classes=2):
    _, base_ctor, _ = get_preprocess_and_base(base_model_name)
    base_model = base_ctor()
    base_model.trainable = False
    model = build_head(base_model, num_classes)
    return model, base_model

# ───────────────────────────── обучение/оценка ──────────────────────────
def compute_class_weights(train_dir, class_names):
    _, labels = collect_files_and_labels(train_dir, class_names)
    cnt = Counter(labels)
    total = sum(cnt.values())
    return {cls: total / (len(cnt) * cnt[cls]) for cls in cnt}

def save_classes_meta(class_names, base_model_name, preprocess_name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    meta = {
        'class_names': class_names,
        'base_model': base_model_name,
        'preprocess': preprocess_name,
        'img_height': IMG_HEIGHT,
        'img_width': IMG_WIDTH
    }
    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_classes_meta():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f'Не найден {CLASSES_PATH}. Нужен для согласованного порядка классов.')
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return meta

def train_model(base_model_name='vgg16', use_class_weights=False, finetune=False, unfreeze=20):
    # авто-классы из train
    class_names = list_classes_from_dir(TRAIN_DIR)
    preprocess_fn, _, preprocess_name = get_preprocess_and_base(base_model_name)
    train_ds, val_ds, test_ds, class_names = create_datasets(base_model_name, class_names)
    num_classes = len(class_names)

    # сохранить метаданные классов и препроцессора
    save_classes_meta(class_names, base_model_name, preprocess_name)

    model, base_model = create_model(base_model_name, num_classes)
    model.compile(optimizer=Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2)
    ]

    class_weight = compute_class_weights(TRAIN_DIR, class_names) if use_class_weights else None

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    if finetune:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze]:
            layer.trainable = False
        model.compile(optimizer=Adam(1e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        ft_callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=2)
        ]
        history_ft = model.fit(
            train_ds,
            epochs=max(3, EPOCHS // 2),
            validation_data=val_ds,
            callbacks=ft_callbacks,
            class_weight=class_weight,
            verbose=1
        )
        for k, v in history_ft.history.items():
            history.history[k] = history.history.get(k, []) + v

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    return model, history, test_ds

def plot_training_history(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='train acc')
    plt.plot(val_acc, label='val acc')
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.title('Loss'); plt.legend()
    plt.tight_layout(); plt.show()

# ───────────────────────────── предсказание ─────────────────────────────
def load_model_safely():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Модель {MODEL_PATH} не найдена. Сначала обучите модель.')
    return keras.models.load_model(MODEL_PATH)

def predict_single_image(model, img_path, base_model_name=None, class_names=None):
    # если не передали — читаем из сохранённых метаданных
    if base_model_name is None or class_names is None:
        meta = load_classes_meta()
        base_model_name = meta['base_model']
        class_names = meta['class_names']

    preprocess_fn, _, _ = get_preprocess_and_base(base_model_name)
    img_bin = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_bin, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    x = preprocess_fn(img)
    x = tf.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    cls_idx = int(np.argmax(probs))
    label = class_names[cls_idx]
    conf = float(probs[cls_idx])
    return label, conf, probs.tolist()

def predict_from_directory(directory_path, base_model_name=None):
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f'Директория {directory_path} не существует')

    meta = load_classes_meta()
    class_names = meta['class_names']
    base_model_name = base_model_name or meta['base_model']

    model = load_model_safely()

    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        image_paths.extend(glob.glob(os.path.join(directory_path, ext)))
        image_paths.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))

    if not image_paths:
        print(f'В директории {directory_path} не найдено изображений')
        return []

    print(f'Найдено {len(image_paths)} изображений для предсказания')
    print('-' * 50)

    results = []
    for p in image_paths:
        try:
            label, conf, raw = predict_single_image(model, p, base_model_name, class_names)
            results.append({'path': p, 'class': label, 'confidence': conf, 'probs': raw})
            print(f'{os.path.basename(p):30} -> {label:20} ({conf:.2%})')
        except Exception as e:
            print(f'Ошибка при обработке {p}: {e}')

    print('-' * 50)
    counts = Counter(r['class'] for r in results)
    for cls in class_names:
        print(f'{cls:20}: {counts.get(cls, 0)}')
    print(f'Общее количество: {len(results)}')
    return results

# ───────────────────────────── CLI ──────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Мультиклассовая классификация с авто-классами из папок')
    parser.add_argument('mode', choices=['train', 'predict'], help='Режим работы: train или predict')
    parser.add_argument('--model', type=str, default='vgg16', help='База: vgg16, resnet50, efficientnet')
    parser.add_argument('--predict-dir', type=str, default=TEST_DIR, help='Папка с изображениями для предсказания')
    parser.add_argument('--class-weights', action='store_true', help='Использовать веса классов')
    parser.add_argument('--finetune', action='store_true', help='Включить тонкую настройку базы')
    parser.add_argument('--unfreeze', type=int, default=20, help='Сколько верхних слоёв базы разморозить при фтюнинге')
    args = parser.parse_args()

    if args.mode == 'train':
        print('Запуск обучения модели...')
        class_names = list_classes_from_dir(TRAIN_DIR)
        check_dataset_integrity(class_names)
        model, history, _ = train_model(
            base_model_name=args.model,
            use_class_weights=args.class_weights,
            finetune=args.finetune,
            unfreeze=args.unfreeze
        )
        plot_training_history(history)
        print('Обучение завершено!')
    elif args.mode == 'predict':
        print(f'Предсказание для изображений в {args.predict_dir}...')
        predict_from_directory(args.predict_dir, base_model_name=args.model)

if __name__ == '__main__':
    main()
