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

# Variable de les mides en les que possarem totes les imatges
target_size = (128, 128)

# Normalitza les imatges i les converteix en tensors ded PyTorch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#Ruta del model entrenat i previament guardat
modelo_svm_ruta = 'svm_model.pkl'

#Conprova si el model esta creat, en cas de que ho estigui es salta aquest pas, ja que tarda molt i així no cal que faci el mateix cada vegada
if not os.path.exists(modelo_svm_ruta):
    print("Entrenando el modelo SVM y guardándolo...")
    
    #fi de ruta de les imatges que volem classificar. La variable url esta definidda a config.
    img_dir_pig = os.path.join(url, "Pig")
    img_dir_dog = os.path.join(url, "Dog")
    img_dir_bread = os.path.join(url, "Bread")

    # Edita el nom dels arxius per afegir-lis la categoria al devant
    img_list_pig = ["pig_" + img for img in os.listdir(img_dir_pig) if img.endswith(".png")]
    img_list_dog = ["dog_" + img for img in os.listdir(img_dir_dog) if img.endswith(".png")]
    img_list_bread = ["bread_" + img for img in os.listdir(img_dir_bread) if img.endswith(".png")]

    # Afegim totes les imatges en una llista
    img_list = img_list_pig + img_list_dog + img_list_bread

    # Cambiem l'ordre de la llista per un altre a l'atzar
    random.shuffle(img_list)

    # Creació de les variables que contindran l'entrenament
    train_images = []
    train_labels = []

    for img in img_list:
        # Separa el tipus del nom original de l'imatge
        type, img = img.split('_', 1)

        # Segons el tipus, assigna un "label"
        if type == "pig":
            label = 0
            img_dir = img_dir_pig
        elif type == "dog":
            label = 1
            img_dir = img_dir_dog
        elif type == "bread":
            label = 2
            img_dir = img_dir_bread

        # processa les imatges
        img_path = os.path.join(img_dir, img)
        img_pil = Image.open(img_path).convert("L")
        img_tensor = transform(img_pil)
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

    # crea un dataset amb les imatges processades
    train_dataset = CustomDataset(train_images, train_labels)

    # Crea un data leader per a facilitar l'acces de les dades
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)

    # redimensiona les imatges
    train_images_resized = [F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0) for img in train_images]

    # Converteix en llistes les matrius NumPy
    X = torch.stack(train_images_resized).numpy()
    y = np.array(train_labels)

    # Separa en variables mes comodes les dades
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Redimensionem les matrius per a que siguin compatibles amb SVM
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Creem i entrenem el classificador SVM
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    clf.fit(X_train, y_train)

    # Guardem el model entrenat per a no haber de fer tot el process sempre
    joblib.dump(clf, 'svm_model.pkl')

    print("Modelo entrenado y guardado exitosamente.")

# Cergem el model creat anteriorment
loaded_model = joblib.load('svm_model.pkl')

# Cargem i processem una imatge
def load_and_process_image(url_img):
    if url_img.endswith(".png"):
        img_pil = Image.open(url_img).convert("L")
        img_tensor = transform(img_pil)
        
        # Redimensiona imatge
        img_resized = F.interpolate(img_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        # redimendiona per a ser compatible amb SVM
        img_resized = img_resized.reshape(1, -1) 
        return img_resized
    else:
        print("La imagen no es png")
        return None

# Funcio que fa que es process l'imatge i llença la predicció
def calcula_que_es(input_image_path):
    # Fem la prediccio
    input_image = load_and_process_image(input_image_path)
    if(input_image != None):
        predicted_class = loaded_model.predict(input_image)
        probabilities = loaded_model.predict_proba(input_image)

        # Mapejem les classes en etiquetes
        class_mapping = {0: "Cerdo", 1: "Perro", 2: "Pan"}
        predicted_label = class_mapping.get(predicted_class[0])

        # Formatejem el percentatge en dos decimals
        formatted_probability = f"{probabilities[0][predicted_class[0]] * 100:.2f}"

        # Retorna el text de la predicció
        return f'La imagen es un.... {predicted_label}, estoy seguro al {formatted_probability} %'
    else: return f'La imagen no tiene formato .png'