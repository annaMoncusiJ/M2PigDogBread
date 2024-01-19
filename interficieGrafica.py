import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
from main import calcula_que_es
import os

def mostrar_imagen(imagen_path, resultado_text):
    global imagen_tk  # Hacer la referencia global

    imagen = Image.open(imagen_path)

    # Obtén las dimensiones reales del lienzo
    canvas_width = lienzo.winfo_reqwidth()
    canvas_height = lienzo.winfo_reqheight()

    # Redimensiona la imagen para ajustarla al lienzo
    imagen.thumbnail((canvas_width, canvas_height))
    imagen_tk = ImageTk.PhotoImage(imagen)

    # Calcula la posición para centrar la imagen en el lienzo
    x = (canvas_width - imagen_tk.width()) // 2
    y = (canvas_height - imagen_tk.height()) // 2

    # Limpia el lienzo antes de mostrar la nueva imagen
    lienzo.delete("all")

    # Muestra la imagen en el lienzo
    lienzo.create_image(x, y, anchor=tk.NW, image=imagen_tk)

    # Actualiza el texto de resultado
    resultado_text.config(text=calcula_que_es(imagen_path))

def on_drop(event):
    archivo = event.data
    archivo_sin_corchetes = archivo.replace('{', '').replace('}', '')  # Eliminar corchetes
    archivo_norm = os.path.normpath(archivo_sin_corchetes)  # Normalizar la ruta
    mostrar_imagen(archivo_norm, resultado)


# Crear la ventana principal
ventana = TkinterDnD.Tk()
ventana.title("Arrastra y Suelta Imagen")

# Crear un lienzo para mostrar la imagen
lienzo = tk.Canvas(ventana, width=400, height=400, bg="white")
lienzo.pack(pady=10)

# Etiqueta para mostrar el resultado
resultado = tk.Label(ventana, text="", font=("Helvetica", 12))
resultado.pack(pady=10)
resultado.config(text="Arrastra y Suelta Imagen")


# Configurar la zona de arrastre (que es el propio lienzo)
lienzo.drop_target_register(DND_FILES)
lienzo.dnd_bind('<<Drop>>', on_drop)

ventana.mainloop()