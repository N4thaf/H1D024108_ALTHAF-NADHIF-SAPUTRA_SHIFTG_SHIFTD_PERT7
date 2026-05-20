import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('iris.data', header=None, sep=',')


print("=== INFO DATASET ===")
print(f"Jumlah data: {len(dataset)}")
print(f"Kolom: {dataset.shape[1]}")
print(dataset.head())


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("\nKelas yang tersedia:", label_encoder.classes_)
print("Contoh label setelah encoding:", y[:10])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nData training: {X_train.shape}")
print(f"Data testing:  {X_test.shape}")


model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])


print("\n=== ARSITEKTUR MODEL ===")
model.summary()


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


print("\n=== PROSES TRAINING ===")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n=== HASIL EVALUASI ===")
print(f"Loss    : {loss:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Training History - Iris Classification', fontsize=14)


axes[0].plot(history.history['loss'], label='Train Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Val Loss', color='orange')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
axes[1].set_title('Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Grafik training history disimpan ke 'training_history.png'")


predictions = model.predict(X_test, verbose=0)
predicted_classes = predictions.argmax(axis=1)

print("\n=== HASIL PREDIKSI ===")
print("Prediksi :", predicted_classes)
print("Label Asli:", y_test)


correct = np.sum(predicted_classes == y_test)
print(f"\nPrediksi benar: {correct}/{len(y_test)}")


cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Iris Classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix disimpan ke 'confusion_matrix.png'")


def predict_new_data():
    print("\n=== PREDIKSI DATA BARU ===")
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width  = float(input("Masukkan sepal width : "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width  = float(input("Masukkan petal width : "))


    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])


    prediction = model.predict(new_data, verbose=0)
    predicted_class = prediction.argmax(axis=1)


    print("\nProbabilitas per kelas:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {cls}: {prediction[0][i]*100:.2f}%")


    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"\n>>> Prediksi kelas: {predicted_label[0]} <<<")


predict_new_data()
