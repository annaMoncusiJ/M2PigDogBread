from config import *
import torch
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import torch.nn.functional as F
from interficieGrafica import *

# Redimensionar todas las imágenes al mismo tamaño
target_size = (128, 128)  # Cambia el tamaño según tus necesidades
# Transformaciones para normalizar las imágenes y convertirlas a tensores de PyTorch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Ruta del modelo entrenado
modelo_svm_ruta = 'svm_model.pkl'
#-------------------------------------
# Ruta base para las carpetas de imágenes
# url = "C:\\Users\\aleej\\Desktop\\IA BIG DATA\\M2\\M2 A5 ClasificadorDogPigBread\\M2DogPigBread\\"
# La url s'agafa de "config.py"
# Verificar si el modelo ya existe
if not os.path.exists(modelo_svm_ruta):
    print("Entrenando el modelo SVM y guardándolo...")
    
    # Ruta de la imagen de entrada que deseas clasificar
    img_dir_pig = os.path.join(url, "Pig")
    img_dir_dog = os.path.join(url, "Dog")
    img_dir_bread = os.path.join(url, "Bread")

    # Listas de nombres de archivos para cada categoría
    img_list_pig = ["pig_" + img for img in os.listdir(img_dir_pig) if img.endswith(".png")]
    img_list_dog = ["dog_" + img for img in os.listdir(img_dir_dog) if img.endswith(".png")]
    img_list_bread = ["bread_" + img for img in os.listdir(img_dir_bread) if img.endswith(".png")]

    # Lista combinada de nombres de archivos
    img_list = img_list_pig + img_list_dog + img_list_bread

    random.shuffle(img_list)

    train_images = []
    train_labels = []

    for img in img_list:
        type, img = img.split('_', 1)

        if type == "pig":
            label = 0
            img_dir = img_dir_pig
        elif type == "dog":
            label = 1
            img_dir = img_dir_dog
        elif type == "bread":
            label = 2
            img_dir = img_dir_bread

        img_path = os.path.join(img_dir, img)
        img_pil = Image.open(img_path).convert("L")
        img_tensor = transform(img_pil)
        train_images.append(img_tensor)
        train_labels.append(label)

    # Crear DataLoader personalizado para cargar las imágenes de entrenamiento
    class CustomDataset(torch.utils.data.Dataset):
        def _init_(self, images, labels):
            self.images = images
            self.labels = labels

        def _len_(self):
            return len(self.images)

        def _getitem_(self, idx):
            return self.images[idx], self.labels[idx]

    train_dataset = CustomDataset(train_images, train_labels)

    # Crear un DataLoader para facilitar el acceso a los datos en lotes
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)


    train_images_resized = [F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0) for img in train_images]

    # Convertir listas a matrices NumPy
    X = torch.stack(train_images_resized).numpy()
    y = np.array(train_labels)

    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Redimensionar las matrices para que sean compatibles con el clasificador SVM
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Crear y entrenar el clasificador SVM
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    clf.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(clf, 'svm_model.pkl')

    print("Modelo entrenado y guardado exitosamente.")
# Cargar el modelo SVM previamente entrenado
loaded_model = joblib.load('svm_model.pkl')

# Función para cargar y procesar una imagen de entrada
def load_and_process_image(url_img):
    if url_img.endswith(".png"):
        img_pil = Image.open(url_img).convert("L")
        img_tensor = transform(img_pil)
        
        # Redimensionar la imatge al mismo tamaño que las imágenes de entrenamiento
        img_resized = F.interpolate(img_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        img_resized = img_resized.reshape(1, -1)  # Redimensionar para que sea compatible con el clasificador SVM
        return img_resized
    else:
        print("La imagen no es png")

def calcula_que_es(input_image_path):
    # Realizar la predicción para la imagen de entrada
    input_image = load_and_process_image(input_image_path)
    predicted_class = loaded_model.predict(input_image)
    probabilities = loaded_model.predict_proba(input_image)

    # Mapear las clases predichas a etiquetas deseadas
    class_mapping = {0: "Cerdo", 1: "Perro", 2: "Pan"}
    predicted_label = class_mapping.get(predicted_class[0])
    # Formatear la probabilidad en porcentaje con dos decimales
    formatted_probability = f"{probabilities[0][predicted_class[0]] * 100:.2f}"

    # Imprimir la clase predicha
    return f'La imagen es un.... {predicted_label}, estoy seguro al {formatted_probability} %'