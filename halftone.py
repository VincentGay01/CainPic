import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cupyx.scipy import ndimage

def halftone_gpu_optimized(image_path, cell=4, max_radius=None):

    if max_radius is None:
        max_radius = cell / 11
    
    # Charger l'image
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    
    # Transférer vers GPU
    img_gpu = cp.asarray(img_array, dtype=cp.float32)
    
    h, w = img_gpu.shape
    
    # Calculer les dimensions de sortie
    out_h = h // cell
    out_w = w // cell
    
    # Redimensionner l'image pour qu'elle soit divisible par cell
    cropped_h = out_h * cell
    cropped_w = out_w * cell
    img_cropped = img_gpu[:cropped_h, :cropped_w]
    
    # Reshaper pour créer les blocs directement
    # (out_h, cell, out_w, cell) puis moyenner sur les axes des cellules
    blocks = img_cropped.reshape(out_h, cell, out_w, cell)
    mean_intensities = cp.mean(blocks, axis=(1, 3)) / 255.0
    
    # Calculer les rayons pour tous les blocs
    radii = (1 - mean_intensities) * max_radius
    
    # Créer les coordonnées des centres
    y_centers, x_centers = cp.mgrid[:out_h, :out_w]
    x_centers = x_centers * cell + cell/2
    y_centers = y_centers * cell + cell/2
    
    # Filtrer les points avec rayon > 0
    mask = radii > 0
    
    # Extraire les coordonnées et tailles valides
    X = cp.extract(mask, x_centers)
    Y = cp.extract(mask, y_centers)
    radii_valid = cp.extract(mask, radii)
    S = (radii_valid * 2) ** 2
    
    # Transférer vers CPU pour matplotlib
    X_cpu = cp.asnumpy(X)
    Y_cpu = cp.asnumpy(Y)
    S_cpu = cp.asnumpy(S)
    
    return X_cpu, Y_cpu, S_cpu


# Version principale avec CuPy
def create_halftone_gpu(image_path="test5.jpg", cell=4, output_path="halftone_gpu.png"):

    X, Y, S = halftone_gpu_optimized(image_path, cell)
    
    # Affichage
    plt.figure(figsize=(8,8))
    plt.scatter(X, -Y, c="black", s=S, marker="o")
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=1)
    plt.show()
    
    print(f"Halftone sauvegardé: {output_path}")
    print(f"Nombre de points: {len(X)}")

# Utilisation
if __name__ == "__main__":

    create_halftone_gpu("flou_surface_interactif.jpg", cell=3)