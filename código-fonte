import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Caminho do dataset
DATASET_PATH = r"c:\Users\User\Downloads\archive"  # Substitua pelo caminho correto

# Função para carregar as imagens e rótulos
def load_images(dataset_path, target_size=(128, 128)):
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if not file.endswith((".jpg", ".jpeg", ".png")):
                print(f"Ignorando arquivo: {file}")
                continue

            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erro ao carregar a imagem: {img_path}")
                continue

            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Carregar o dataset
print("Carregando o dataset...")
images, labels = load_images(DATASET_PATH)

# Normalizar as imagens (0 a 1)
images = images / 255.0

# Codificar rótulos para valores numéricos
unique_labels = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
numeric_labels = np.array([label_to_index[label] for label in labels])

# One-hot encoding dos rótulos
categorical_labels = to_categorical(numeric_labels, num_classes=len(unique_labels))

# Dividir em treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)

print(f"Imagens carregadas: {images.shape[0]}")
print(f"Classes encontradas: {len(unique_labels)}")
print(f"Shape do conjunto de treinamento: {X_train.shape}")
print(f"Shape do conjunto de validação: {X_val.shape}")

# Criar o modelo CNN
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Instanciar o modelo
input_shape = X_train.shape[1:]  # (128, 128, 3)
num_classes = len(unique_labels)
model = create_model(input_shape, num_classes)

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Configurar Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinar o modelo
print("Treinando o modelo...")
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=25, 
                    batch_size=32, 
                    callbacks=[early_stopping])

# Avaliar o modelo
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Acurácia na validação: {val_accuracy * 100:.2f}%")

# Salvar o modelo treinado
model.save("face_classification_model.h5")
print("Modelo salvo como 'face_classification_model.h5'.")

# Analisar resultados
import matplotlib.pyplot as plt

# Plotar a perda
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotar a acurácia
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
