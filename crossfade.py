import imageio.v2 as imageio
import os

frames_dir = "frames_threads_t8"
output_video = "crossfade_openmp_t8.mp4"
fps = 24

frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

with imageio.get_writer(output_video, fps=fps) as writer:
    for frame_name in frames:
        frame_path = os.path.join(frames_dir, frame_name)
        image = imageio.imread(frame_path)
        writer.append_data(image)

print(f"Video generado: {output_video}")