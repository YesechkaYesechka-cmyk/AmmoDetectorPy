import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ────────────────────────────── НАСТРОЙКИ ──────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = 'cats_vs_dogs_model.h5'

TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'
TEST_DIR = 'data/test'

CLASS_NAMES = ['cats', 'dogs']  # жёстко фиксируем порядок для интерпретации сигмоиды

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

# ───────────── вспомогательные: выбор базы и препроцессинга ────────────
def get_preprocess_and_base(base_model_name: str):
    name = base_model_name.lower()
    if name == 'vgg16':
        preprocess = vgg16_preprocess
        base_ctor = lambda: VGG16(weights='imagenet', include_top=False,
                                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    elif name == 'resnet50':
        preprocess = resnet50_preprocess
        base_ctor = lambda: ResNet50(weights='imagenet', include_top=False,
                                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    elif name == 'efficientnet':
        preprocess = effnet_preprocess
        base_ctor = lambda: EfficientNetB0(weights='imagenet', include_top=False,
                                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    else:
        raise ValueError('Неизвестная базовая модель: используйте vgg16, resnet50 или efficientnet')

    def resize_with_pad_and_preprocess(x):
        # x: tf.Tensor [H, W, C] или [B, H, W, C], dtype float32|uint8|int
        x = tf.cast(x, tf.float32)
        if x.shape.rank == 3:
            x = tf.image.resize_with_pad(x, IMG_HEIGHT, IMG_WIDTH)
            x = preprocess(x)
        elif x.shape.rank == 4:
            x = tf.image.resize_with_pad(x, IMG_HEIGHT, IMG_WIDTH)
            x = preprocess(x)
        else:
            raise ValueError('Ожидается изображение ранга 3 или 4')
        return x

    return resize_with_pad_and_preprocess, base_ctor

# ───────────── чтение датасета через tf.data без потери пропорций ───────
def collect_files_and_labels(root_dir: str, class_names=CLASS_NAMES):
    filepaths = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
            filepaths.extend(glob.glob(os.path.join(cls_dir, ext)))
            filepaths.extend(glob.glob(os.path.join(cls_dir, '**', ext), recursive=True))
        labels.extend([idx] * (len(filepaths) - len(labels)))
    return filepaths, labels

def make_dataset(root_dir: str, preprocess_fn, shuffle: bool, augment: bool):
    filepaths, labels = collect_files_and_labels(root_dir)
    if len(filepaths) == 0:
        raise FileNotFoundError(f'В {root_dir} не найдено изображений в папках {CLASS_NAMES}')

    paths_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.float32))

    def load_image(path):
        img_bin = tf.io.read_file(path)
        img = tf.io.decode_image(img_bin, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return img

    # аугментации только для train
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
        img = preprocess_fn(img)  # resize_with_pad + preprocess_input
        return img, tf.expand_dims(label, axis=-1)  # бинарная метка (B,1)

    ds = tf.data.Dataset.zip((paths_ds, labels_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2048, len(filepaths)), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def create_datasets(base_model_name='vgg16'):
    preprocess_fn, _ = get_preprocess_and_base(base_model_name)
    train_ds = make_dataset(TRAIN_DIR, preprocess_fn, shuffle=True, augment=True)
    val_ds   = make_dataset(VALIDATION_DIR, preprocess_fn, shuffle=False, augment=False)
    test_ds  = make_dataset(TEST_DIR, preprocess_fn, shuffle=False, augment=False)
    return train_ds, val_ds, test_ds

# ───────────────────────────── проверка данных ───────────────────────────
def check_dataset_integrity():
    print('Проверка целостности данных...')
    for cls in CLASS_NAMES:
        counts = {}
        for split, root in [('train', TRAIN_DIR), ('val', VALIDATION_DIR), ('test', TEST_DIR)]:
            dir_path = os.path.join(root, cls)
            cnt = 0
            if os.path.isdir(dir_path):
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
                    cnt += len(glob.glob(os.path.join(dir_path, ext)))
                    cnt += len(glob.glob(os.path.join(dir_path, '**', ext), recursive=True))
            counts[split] = cnt
        print(f'{cls}: train={counts["train"]}, val={counts["val"]}, test={counts["test"]}')

# ───────────────────────────── построение модели ────────────────────────
def create_model(base_model_name='vgg16'):
    _, base_ctor = get_preprocess_and_base(base_model_name)
    base_model = base_ctor()
    base_model.trainable = False  # первый этап — только «голова»

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),  # бинарная классификация: dog=1, cat=0
    ])
    return model

# ───────────────────────────── обучение/оценка ──────────────────────────
def train_model(base_model_name='vgg16'):
    train_ds, val_ds, test_ds = create_datasets(base_model_name)
    model = create_model(base_model_name)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2)
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    print(f'Сохраняю модель в {MODEL_PATH}')
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

    plt.figure(figsize=(12,4))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(acc, label='train acc')
    plt.plot(val_acc, label='val acc')
    plt.title('Accuracy')
    plt.legend()
    # Loss
    plt.subplot(1,2,2)
    plt.plot(loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ───────────────────────────── предсказание ─────────────────────────────
def load_model_safely():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Модель {MODEL_PATH} не найдена. Сначала обучите модель.')
    return keras.models.load_model(MODEL_PATH)

def predict_single_image(model, img_path, base_model_name='vgg16'):
    preprocess_fn, _ = get_preprocess_and_base(base_model_name)
    img_bin = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_bin, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    x = preprocess_fn(img)                # resize_with_pad + preprocess_input
    x = tf.expand_dims(x, axis=0)         # [1, H, W, C]
    pred = model.predict(x, verbose=0)[0][0]
    label = 'Собака' if pred > 0.5 else 'Кошка'
    conf = float(pred if pred > 0.5 else 1 - pred)
    return label, conf, float(pred)

def predict_from_directory(directory_path, base_model_name='vgg16'):
    model = load_model_safely()
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f'Директория {directory_path} не существует')

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
            label, conf, raw = predict_single_image(model, p, base_model_name)
            results.append({'path': p, 'class': label, 'confidence': conf, 'raw_prediction': raw})
            print(f'{os.path.basename(p):20} -> {label:6} ({conf:.2%})')
        except Exception as e:
            print(f'Ошибка при обработке {p}: {e}')
    return results

# ───────────────────────────── CLI ──────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Классификация кошек и собак с сохранением пропорций')
    parser.add_argument('mode', choices=['train', 'predict'], help='Режим работы: train или predict')
    parser.add_argument('--model', type=str, default='vgg16', help='База: vgg16, resnet50, efficientnet')
    parser.add_argument('--predict-dir', type=str, default='data/test', help='Папка с изображениями для предсказания')
    args = parser.parse_args()

    if args.mode == 'train':
        print('Запуск обучения модели...')
        check_dataset_integrity()
        model, history, test_ds = train_model(args.model)
        plot_training_history(history)
        print('Обучение завершено!')
    elif args.mode == 'predict':
        print(f'Предсказание для изображений в {args.predict_dir}...')
        preds = predict_from_directory(args.predict_dir, args.model)
        if preds:
            cats = sum(1 for p in preds if p['class'] == 'Кошка')
            dogs = sum(1 for p in preds if p['class'] == 'Собака')
            print('-' * 50)
            print(f'Статистика: Кошки - {cats}, Собаки - {dogs}')
            print(f'Общее количество: {len(preds)}')

if __name__ == '__main__':
    main()
