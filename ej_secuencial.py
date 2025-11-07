from PIL import Image
import numpy as np
import time
import argparse
import os

NUM_FRAMES = 96  # 4 segundos a 24 FPS
INPUT_FILENAME = "image-800.jpg"
OUTPUT_DIR = "frames_secuencial"

# crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_to_grayscale(original_image: Image.Image) -> Image.Image:
    #convertir una imagen RGB a escala de grises usando NumPy
    arr = np.array(original_image, dtype=np.float32)

    # formula de luminancia: G = 0.299R + 0.587G + 0.114B
    # Para convertir una imagen de color a una que solo use tonos de gris,
    # se necesita encontrar un unico valor de brillo que sea percibido por el ojo humano
    # como equivalente al color original.
    # fuente: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
    gray_vals = (0.299 * arr[:, :, 0] +
                 0.587 * arr[:, :, 1] +
                 0.114 * arr[:, :, 2]).astype(np.uint8) #convierte los valores de vuelta a enteros entre 0–255

    # replicar el canal gris en R,G,B
    gray_arr = np.stack((gray_vals, gray_vals, gray_vals), axis=-1) #axis=-1 apila al final

    # convertir de nuevo a imagen PIL
    return Image.fromarray(gray_arr, mode="RGB")


def generate_cross_fading_secuencial(image1_color: Image.Image, image2_gray: Image.Image, args):
    
    #genera los frames de transición entre la imagen color y su versión en grises.
    #Todo el calculo se hace en NumPy (sin bucles).
    
    img1_arr = np.array(image1_color, dtype=np.float32)
    img2_arr = np.array(image2_gray, dtype=np.float32)

    for i in range(NUM_FRAMES):
        P = 1.0 - (i / (NUM_FRAMES - 1))
        P_comp = 1.0 - P

        # mezcla lineal de ambas imagenes (todo el array a la vez)
        frame_arr = (img1_arr * P + img2_arr * P_comp).astype(np.uint8)

        # convertir a imagen y guardar
        if not args.no_save:
            frame_img = Image.fromarray(frame_arr, mode="RGB")
            filename = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
            frame_img.save(filename)
            print(f"Guardado: {filename} (P = {P:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Secuencial.")
    parser.add_argument("--no-save", action="store_true", help="Si está activado, no guardar frames.")
    args = parser.parse_args()
    # carga de imagen
    try:
        color_img = Image.open(INPUT_FILENAME).convert("RGB")
    except Exception as e:
        print(f"Error: No se pudo cargar la imagen {INPUT_FILENAME}")
        print(e)
        return

    print(f"Imagen cargada. Dimensiones: {color_img.size[0]}x{color_img.size[1]}")

    # conversion a grises
    print("Generando la imagen en escala de grises (NumPy)...")
    gray_img = convert_to_grayscale(color_img)

    # generación secuencial de frames
    print(f"Iniciando generación de {NUM_FRAMES} frames (SECUENCIAL con NumPy)...")
    start_time = time.time()

    generate_cross_fading_secuencial(color_img, gray_img, args)

    elapsed_time = time.time() - start_time

    print("\n=======================================================")
    print(f"Tiempo de ejecución SECUENCIAL (96 frames, NumPy): {elapsed_time:.3f} segundos")
    print("=======================================================")


if __name__ == "__main__":
    main()

# commando para correr: python3 ej_secuencial.py --no-save