import tensorflow as tf

# если у вас .h5
model = tf.keras.models.load_model('ammo_model.h5')
# (лучше сохранить ещё и в новом формате)
model.save('ammo_model.keras')

# Экспорт в TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Опционально: небольшая оптимизация (быстрее, размер меньше)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('ammo_model.tflite', 'wb') as f:
    f.write(tflite_model)

print('Saved cats_vs_dogs_model.tflite')
