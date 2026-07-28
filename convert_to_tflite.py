import tensorflow as tf

# Загружаем модель
model = tf.keras.models.load_model('models/enhancer_model.keras')

# Конвертируем в TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Сохраняем
with open('models/enhancer_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Модель сконвертирована в TFLite")
print(f"Размер: {len(tflite_model) / 1024:.2f} КБ")