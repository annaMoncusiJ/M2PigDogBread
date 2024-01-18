import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np # Importar NumPy
import os
from PIL import Image
import re
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Transformaciones para normalizar las imágenes y convertirlas a tensores de PyTorch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
url = "C:\\Users\\Francesc\\Downloads\\M4DogPigBread\\"
# Cargar el conjunto de datos MNIST desde "Descargas\img" y aplicar transformaciones
img_dir_pig = url+"Pig"
img_dir_dog = url+"Dog"
img_dir_bread = url+"Bread"
img_list_pig = ["pig_" + img for img in os.listdir(img_dir_pig)] 
img_list_dog = ["dog_" + img for img in os.listdir(img_dir_dog)] 
img_list_bread = ["bread_" + img for img in os.listdir(img_dir_bread)] 
img_list = img_list_pig + img_list_dog + img_list_bread

random.shuffle(img_list)

train_images = []
train_labels = []

for img in img_list:
    if img.endswith(".png"): # Asumiendo que las imágenes son archivos .jpg
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
            
        # label = int(img.split("_")[0]) # Extraer la etiqueta de la imagen
        img_path = os.path.join(img_dir, img)
        img_pil = Image.open(img_path).convert("L") # Convertir a escala de grises
        img_tensor = transform(img_pil) # Aplicar transformaciones
        train_images.append(img_tensor)
        train_labels.append(label)

# Crear DataLoader personalizado para cargar las imágenes de entrenamiento
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_dataset = CustomDataset(train_images, train_labels)

# Crear un DataLoader para facilitar el acceso a los datos en lotes
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)


# # Iterar a través del DataLoader
# for images, labels in train_loader:
#     # Visualizar algunas de las imágenes del lote
#     def imshow(img):
#         img = img / 2 + 0.5  # Desnormalizar
#         np_img = img.numpy()
#         plt.imshow(np.transpose(np_img, (1, 2, 0)))
#         plt.show()

#     # Mostrar imágenes y etiquetas
#     imshow(torchvision.utils.make_grid(images))
#     print('Etiquetas:', ' '.join(str(labels[j].item()) for j in range(5)))
    
#     # Romper el bucle después de visualizar un lote
#     break


# Convertir listas a matrices NumPy
X = np.array(train_images)
y = np.array(train_labels)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el clasificador SVM
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')