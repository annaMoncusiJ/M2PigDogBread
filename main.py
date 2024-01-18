import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np # Importar NumPy
import os
from PIL import Image
import re
import random


# Transformaciones para normalizar las imágenes y convertirlas a tensores de PyTorch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Cargar el conjunto de datos MNIST desde "Descargas\img" y aplicar transformaciones
img_dir = "C:\\Users\\Francesc\\Downloads\\ImgTools.co-PNG"
img_list = os.listdir(img_dir)

train_images = []
train_labels = []

for img in img_list:
    if img.endswith(".png"): # Asumiendo que las imágenes son archivos .jpg
        numbers = [int(digit) for digit in re.findall(r'\d', img)]
        label = ''.join(str(num) for num in numbers)
        if(label == ""):
            label = random.randint(10000, 9999999999)

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