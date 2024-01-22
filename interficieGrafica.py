import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
from main import calcula_que_es
import os

def mostrar_imagen(imagen_path, resultado_text):
    global imagen_tk

    imagen = Image.open(imagen_path)

    # Obte les dimencions de la xona on es posa l'imatge
    canvas_width = lienzo.winfo_reqwidth()
    canvas_height = lienzo.winfo_reqheight()

    # Redimensiona la imagen per ajustarla a l'interficie grafica
    imagen.thumbnail((canvas_width, canvas_height))
    imagen_tk = ImageTk.PhotoImage(imagen)

    # Calcula la posició per a centrar l'imatge
    x = (canvas_width - imagen_tk.width()) // 2
    y = (canvas_height - imagen_tk.height()) // 2

    # Elimina l'imatge anterior
    lienzo.delete("all")

    # Mostra l'imatge
    lienzo.create_image(x, y, anchor=tk.NW, image=imagen_tk)

    # Canvia el text amb el nou resultat
    resultado_text.config(text=calcula_que_es(imagen_path))

# crea un event
def on_drop(event):
    #agafa les dades de la foto penjada
    archivo = event.data
    
    # ho torna en text pla per a que no dongui errors amb els espais
    archivo_sin_corchetes = archivo.replace('{', '').replace('}', '')
    archivo_norm = os.path.normpath(archivo_sin_corchetes)

    mostrar_imagen(archivo_norm, resultado)


# Crea la finestra principal de l'interficie grafica
ventana = TkinterDnD.Tk()
ventana.title("Arrastra y Suelta Imagen")

# Crear un "lienzo" par a mostrar l'imatge'
lienzo = tk.Canvas(ventana, width=400, height=400, bg="white")
lienzo.pack(pady=10)

# Crea l'etiqueta on es mostrara el text
resultado = tk.Label(ventana, text="", font=("Helvetica", 12))
resultado.pack(pady=10)
resultado.config(text="Arrastra y Suelta Imagen")


# Configura per a que es puguin arrosegar les imatges
lienzo.drop_target_register(DND_FILES)
lienzo.dnd_bind('<<Drop>>', on_drop)

# obra la finestra
ventana.mainloop()