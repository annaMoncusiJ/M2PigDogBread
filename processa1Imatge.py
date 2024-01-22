url_img = "C:\\Users\\Francesc\\Downloads\\M4DogPigBread\\Bread\\1500.png"


imatge_processada = ""

if url_img.endswith(".png"): # Asumiendo que las imágenes son archivos .jpg
    img = url_img

    # label = int(img.split("_")[0]) # Extraer la etiqueta de la imagen
    img_path = os.path.join(url_img, img)
    img_pil = Image.open(img_path).convert("L") # Convertir a escala de grises
    imatge_processada = transform(img_pil) # Aplicar transformaciones
else:
    print("la imatge no és png")