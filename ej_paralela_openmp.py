#!/usr/bin/env python3
"""
estilo openMP usando threads + numpy
se divide la imagen por filas, cad athear brocesa su bloque
hace el crossfading para los frames y el maestro es quien guarda los frames
"""

from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import math
import argparse
from multiprocessing import cpu_count #detecta cuantos nucleos logicos tiene el procesador

NUM_FRAMES = 96
INPUT_FILENAME = "image-5000.jpeg"
OUTPUT_DIR_BASE = "frames_threads" 

def convert_to_grayscale_array_rgb(color_arr: np.ndarray) -> np.ndarray:
    arr_f = color_arr.astype(np.float32)
    gray_vals = (0.299 * arr_f[:, :, 0] +
                 0.587 * arr_f[:, :, 1] +
                 0.114 * arr_f[:, :, 2])
    gray_u8 = np.clip(gray_vals, 0, 255).astype(np.uint8)
    gray_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=2)
    return gray_rgb

def blend_chunk(start_row: int, end_row: int,
                img1_arr: np.ndarray, img2_arr: np.ndarray,
                result_arr: np.ndarray,  #array compartido donde se escribe el resultado
                P: float):

    if start_row >= end_row:
        return  
    a = img1_arr[start_row:end_row].astype(np.float32)
    b = img2_arr[start_row:end_row].astype(np.float32)
    blended = a * P + b * (1.0 - P)
    np.copyto(result_arr[start_row:end_row], blended.clip(0, 255).astype(np.uint8))

def run_with_threads(num_threads: int, color_img: Image.Image, out_dir: str, args):
    os.makedirs(out_dir, exist_ok=True)

    color_arr = np.array(color_img.convert("RGB"), dtype=np.uint8)
    height, width, _ = color_arr.shape

    gray_arr = convert_to_grayscale_array_rgb(color_arr)

    # prepara el array destino
    result_arr = np.empty_like(color_arr)

    # Precalcular cortes por thread (división por filas)
    rows_per_thread = height // num_threads
    remainder = height % num_threads
    # lista de (start, end)
    slices = []
    start = 0
    for t in range(num_threads):
        extra = 1 if t < remainder else 0
        block_rows = rows_per_thread + extra
        end = start + block_rows
        slices.append((start, end))
        start = end

    t0 = time.perf_counter()

    # Para cada frame, lanzamos tareas a los threads para calcular sus porciones
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for frame_idx in range(NUM_FRAMES):
            P = 1.0 - (frame_idx / (NUM_FRAMES - 1))
            # lanzar todas las tareas (cada tarea mezcla su slice)
            futures = []
            for start_row, end_row in slices:
                futures.append(executor.submit(blend_chunk,
                                               start_row, end_row,
                                               color_arr, gray_arr,
                                               result_arr, P))
            # esperar a que terminen todas 
            for f in as_completed(futures):
                # re-raise si hubo excepción
                exc = f.exception()
                if exc:
                    raise exc

            # Guardado (I/O) — realizado por el hilo principal
            if not args.no_save:
                frame_img = Image.fromarray(result_arr, mode="RGB")
                filename = os.path.join(out_dir, f"frame_{frame_idx:03d}.png")
                frame_img.save(filename)
            # frame_img.save(filename)

    t1 = time.perf_counter()
    return t1 - t0

def main():
    parser = argparse.ArgumentParser(description="Simular OpenMP con threads en Python (RGB).")
    parser.add_argument("--no-save", action="store_true", help="Si está activado, no guardar frames.")
    parser.add_argument("--threads", "-t", type=int, nargs="*", default=None,
                        help="Lista de counts de threads a probar. Ej: -t 1 2 4. Si no se pasa, usa [1, cpu_count()].")
    args = parser.parse_args()


    try:
        color_img = Image.open(INPUT_FILENAME).convert("RGB")
    except Exception as e:
        print(f"Error cargando imagen {INPUT_FILENAME}: {e}")
        return

    # determinar configuraciones de threads
    max_cpu = cpu_count()
    if args.threads is None or len(args.threads) == 0:
        thread_list = [1, min(2, max_cpu), min(4, max_cpu), max_cpu]
        thread_list = sorted(set([t for t in thread_list if t > 0]))
    else:
        thread_list = sorted(set([t for t in args.threads if t > 0 and t <= max_cpu]))

    print(f"Imagen: {INPUT_FILENAME} ({color_img.size[0]}x{color_img.size[1]}), CPU logical: {max_cpu}")
    print(f"Probando configuraciones de threads: {thread_list}\n")

    results = {}
    for nt in thread_list:
        out_dir = f"{OUTPUT_DIR_BASE}_t{nt}"
        print(f"-> Ejecutando con {nt} threads. Output: {out_dir}")
        elapsed = run_with_threads(nt, color_img, out_dir, args)
        results[nt] = elapsed
        print(f"   Tiempo total (no incluye guardado) con {nt} threads: {elapsed:.3f} s\n")

    print("Resumen:")
    for nt, t in results.items():
        print(f"  {nt:2d} threads -> {t:.3f} s")

if __name__ == "__main__":
    main()

# COMANDO python3 ej_paralela_openmp.py -t 1 2 4 8 --no-save