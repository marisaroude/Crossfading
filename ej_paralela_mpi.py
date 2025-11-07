#!/usr/bin/env python3
"""
ej_paralela_mpi.py

 MPI (memoria distribuida)
 - divide la imagen por filas entre procesos (Scatterv)
 - cada proceso convierte SU porción a escala de grises 
 - cada proceso calcula SU porción del cross-fading 
 - el root reúne (Gatherv) y guarda cada frame
"""

import argparse
import os
import time
from typing import Tuple

from PIL import Image
import numpy as np
from mpi4py import MPI


def compute_sendcounts_displs(height: int, width: int, nproc: int, channels: int = 3) -> Tuple[list, list]:
    rows_per = height // nproc
    rem = height % nproc
    sendcounts = [] #cantidad de bytes a cada proceso
    displs = [] #displacements
    offset = 0
    for i in range(nproc):
        rows = rows_per + (1 if i < rem else 0)
        cnt = rows * width * channels
        sendcounts.append(cnt)
        displs.append(offset)
        offset += cnt
    return sendcounts, displs


def convert_rgb_block_to_gray_block(local_color_block: np.ndarray) -> np.ndarray:
    if local_color_block.size == 0:
        return local_color_block.reshape((0, local_color_block.shape[1], 3))
    arr_f = local_color_block.astype(np.float32)
    gray_vals = (0.299 * arr_f[:, :, 0] + 0.587 * arr_f[:, :, 1] + 0.114 * arr_f[:, :, 2])
    gray_u8 = np.clip(gray_vals, 0, 255).astype(np.uint8)
    gray_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=2)  # (H_local, W, 3)
    return gray_rgb


# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paralela con MPI.")
    parser.add_argument("--no-save", action="store_true", help="Si está activado, no guardar frames.")
    args = parser.parse_args()

    INPUT_FILENAME = "image-2000.jpeg"
    NUM_FRAMES = 96
    OUTPUT_DIR = "frames_mpi"

    # MPI init
    comm = MPI.COMM_WORLD #grupo de todos los procesos
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"[rank 0] procesos MPI: {size}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Root carga la imagen y prepara los buffers aplanados.
    width = 0
    height = 0
    color_flat = None  # 1D uint8 array on root
    total_bytes = 0

    if rank == 0:
        try:
            img = Image.open(INPUT_FILENAME).convert("RGB")
        except Exception as e:
            print(f"[rank 0] Error al cargar la imagen {INPUT_FILENAME}: {e}")
            comm.Abort(1)

        width, height = img.size
        print(f"[rank 0] Imagen cargada {INPUT_FILENAME} size={width}x{height}")
        color_arr = np.array(img, dtype=np.uint8)  #  (H, W, 3) 3d
        total_bytes = width * height * 3 #cantidad total de elementos que hay en el buffer plano
        color_flat = color_arr.ravel() # aplana la imagen a 1d, secuencia continua de bytes
        
    # enviar las dimensiones a todos los procesos
    width = comm.bcast(width if rank == 0 else None, root=0)
    height = comm.bcast(height if rank == 0 else None, root=0)

    if size == 1:
        if rank == 0:
            print("[rank 0] Se necesitan mas de un proceso para usar MPI .")
        MPI.Finalize()
        return

    # calcular sendcounts y displacements (en bytes/elementos uint8)
    sendcounts, displs = compute_sendcounts_displs(height, width, size, channels=3)

    # cada rango calcula su numero de filas y el tamaño del búfer local.
    # debemos calcular `my_rows` con la misma lógica que `compute_sendcounts_displs`:
    base_rows = height // size
    rem = height % size
    my_rows = base_rows + (1 if rank < rem else 0)
    my_data_size = my_rows * width * 3  # elements (uint8)

    # asignar buffers locales (planos) para recibir a traves de Scatterv
    local_color_flat = np.empty(my_data_size, dtype=np.uint8)

    #Y el root reparte
    comm.Scatterv([color_flat, sendcounts, displs, MPI.UNSIGNED_CHAR] if rank == 0 else None, local_color_flat, root=0)

    # Transformar local_color en un bloque 3D para su procesamiento.
    if my_rows > 0:
        local_color = local_color_flat.reshape((my_rows, width, 3))
    else:
        # empty block
        local_color = np.zeros((0, width, 3), dtype=np.uint8)

    # ----- Conversión paralela a escala de grises (cada rango procesa su bloque local).-----
    comm.Barrier() #sincroniza para empezar la medición al mismo tiempo.
    t_gray_start = MPI.Wtime()

    local_gray = convert_rgb_block_to_gray_block(local_color)  # (my_rows, width, 3), uint8

    t_gray_end = MPI.Wtime()
    gray_time_local = t_gray_end - t_gray_start
    # simplificamos para obtener un resumen sobre la raíz.
    gray_time_max = comm.reduce(gray_time_local, op=MPI.MAX, root=0)
    gray_time_sum = comm.reduce(gray_time_local, op=MPI.SUM, root=0)

    if rank == 0:
        gray_time_avg = gray_time_sum / size
        print(f"[rank 0] Escala de grises paralela realizada max_time={gray_time_max:.6f}s avg_time={gray_time_avg:.6f}s")

   
    # Preparar búferes para la captura de frames
    # root necesita result_flat para capturar la imagen completa por frames.
    result_flat = None
    if rank == 0:
        result_flat = np.empty(width * height * 3, dtype=np.uint8) # para reunir lsa porciones y reconstruir la imagen

    # asignar local_result_flat
    local_result_flat = np.empty(my_data_size, dtype=np.uint8) #donde cada proceso pondra su porcion procesada

    # Bucle de frames principales (cada rango calcula su mezcla local y luego usa Gatherv para obtener la raíz)    
    comm.Barrier()
    t_all_start = MPI.Wtime()

    for frame_idx in range(NUM_FRAMES):
        P = 1.0 - (frame_idx / (NUM_FRAMES - 1))
        # calcular bloque local combinado (vectorizado)
        if my_rows > 0:
            blended = (local_color.astype(np.float32) * P + local_gray.astype(np.float32) * (1.0 - P))
            local_result = blended.clip(0, 255).astype(np.uint8)
            local_result_flat[:] = local_result.ravel()
        else:
            local_result_flat = np.empty(0, dtype=np.uint8)

        # Reúne todos los resultados locales en result_flat en la raíz.
        comm.Gatherv(local_result_flat, [result_flat, sendcounts, displs, MPI.UNSIGNED_CHAR], root=0)

        # root guarda elframe
        if rank == 0 and not args.no_save:
            full_arr = result_flat.reshape((height, width, 3))
            img_out = Image.fromarray(full_arr, mode="RGB")
            fname = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:03d}.png")
            img_out.save(fname)
         
            if (frame_idx + 1) % max(1, NUM_FRAMES // 10) == 0 or frame_idx == 0 or frame_idx == NUM_FRAMES - 1:
                print(f"[rank 0] Frame guardado {frame_idx:03d} (P={P:.3f})")

    comm.Barrier()
    t_all_end = MPI.Wtime()

    # reporte de los tiempos
    total_time = t_all_end - t_all_start
    total_time_allreduce_max = comm.reduce(total_time, op=MPI.MAX, root=0)
    if rank == 0:
        print("\n=======================================================")
        print(f"[rank 0] Tiempo de bucle de frames (incluyendo Gatherv (sin guardado): {total_time_allreduce_max:.6f} s")
        print("=======================================================")

    if rank == 0:
        print(f"[rank 0]Resumen: escala de grises (paralela) max={gray_time_max:.6f}s avg={gray_time_avg:.6f}s")

    MPI.Finalize()


if __name__ == "__main__":
    main()

# commando ejecucion mpirun -np 4 python3 ej_paralela_mpi.py --no-save 