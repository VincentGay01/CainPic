import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cupyx.scipy import ndimage

def halftone_gpu_optimized_fullres(image_path, cell=4, max_radius=None, cells_per_pixel=1):
    """
    Crée un halftone en préservant la résolution de l'image d'entrée
    
    Parameters:
    - image_path: chemin vers l'image
    - cell: taille de la cellule de base
    - max_radius: rayon maximum des points
    - cells_per_pixel: nombre de cellules par pixel (pour plus de détails)
    """
    
    if max_radius is None:
        max_radius = cell / 2.2
    
    # Charger l'image
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    
    # Transférer vers GPU
    img_gpu = cp.asarray(img_array, dtype=cp.float32)
    
    h, w = img_gpu.shape
    
    # Ajuster la taille de cellule effective
    effective_cell = cell // cells_per_pixel
    if effective_cell < 1:
        effective_cell = 1
    
    # Calculer les dimensions des blocs
    out_h = h // effective_cell
    out_w = w // effective_cell
    
    # Redimensionner l'image pour qu'elle soit divisible par effective_cell
    cropped_h = out_h * effective_cell
    cropped_w = out_w * effective_cell
    img_cropped = img_gpu[:cropped_h, :cropped_w]
    
    # Reshaper pour créer les blocs
    blocks = img_cropped.reshape(out_h, effective_cell, out_w, effective_cell)
    mean_intensities = cp.mean(blocks, axis=(1, 3)) / 255.0
    
    # Calculer les rayons basés sur l'intensité (plus sombre = plus gros point)
    radii = (1 - mean_intensities) * max_radius
    
    # Créer une image de sortie haute résolution
    output_img = cp.ones((h, w), dtype=cp.float32) * 255  # Fond blanc
    
    # Pour chaque bloc, placer des points
    for i in range(out_h):
        for j in range(out_w):
            if radii[i, j] > 0.1:  # Seuil minimum pour éviter les points trop petits
                # Position du centre du bloc dans l'image originale
                center_y = i * effective_cell + effective_cell // 2
                center_x = j * effective_cell + effective_cell // 2
                
                # Créer un motif de points selon cells_per_pixel
                for dy in range(cells_per_pixel):
                    for dx in range(cells_per_pixel):
                        # Position du point dans la sous-grille
                        point_y = center_y + (dy - cells_per_pixel//2) * effective_cell // cells_per_pixel
                        point_x = center_x + (dx - cells_per_pixel//2) * effective_cell // cells_per_pixel
                        
                        # S'assurer que les coordonnées sont dans les limites
                        if 0 <= point_y < h and 0 <= point_x < w:
                            # Créer un cercle autour de ce point
                            radius = radii[i, j] / cells_per_pixel
                            y_min = max(0, int(point_y - radius))
                            y_max = min(h, int(point_y + radius) + 1)
                            x_min = max(0, int(point_x - radius))
                            x_max = min(w, int(point_x + radius) + 1)
                            
                            # Créer les coordonnées pour le cercle
                            y_coords, x_coords = cp.ogrid[y_min:y_max, x_min:x_max]
                            distances = cp.sqrt((x_coords - point_x)**2 + (y_coords - point_y)**2)
                            
                            # Appliquer le cercle (noir sur blanc)
                            mask = distances <= radius
                            output_img[y_min:y_max, x_min:x_max][mask] = 0
    
    return cp.asnumpy(output_img)

def halftone_gpu_grid_based(image_path, cell_size=4, max_radius=None):
    """
    Version grille optimisée sans map_coordinates
    """
    
    if max_radius is None:
        max_radius = cell_size / 2.2
    
    # Charger l'image
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    h, w = img_array.shape
    
    # Transférer vers GPU
    img_gpu = cp.asarray(img_array, dtype=cp.float32)
    
    # Calculer le nombre de cellules
    grid_h = h // cell_size
    grid_w = w // cell_size
    
    # Redimensionner pour être divisible par cell_size
    cropped_h = grid_h * cell_size
    cropped_w = grid_w * cell_size
    img_cropped = img_gpu[:cropped_h, :cropped_w]
    
    # Créer la grille de points
    y_indices = cp.arange(grid_h) * cell_size + cell_size // 2
    x_indices = cp.arange(grid_w) * cell_size + cell_size // 2
    
    # Meshgrid pour avoir toutes les coordonnées
    X_grid, Y_grid = cp.meshgrid(x_indices, y_indices)
    
    # Calculer les intensités moyennes pour chaque cellule
    # Reshaper l'image en blocs
    blocks = img_cropped.reshape(grid_h, cell_size, grid_w, cell_size)
    mean_intensities = cp.mean(blocks, axis=(1, 3)) / 255.0
    
    # Calculer les rayons basés sur l'intensité
    radii = (1 - mean_intensities) * max_radius
    
    # Filtrer les points avec rayon significatif
    mask = radii > 0.1
    
    # Extraire les coordonnées et tailles valides
    X_valid = X_grid[mask]
    Y_valid = Y_grid[mask]
    radii_valid = radii[mask]
    S_valid = (radii_valid * 2) ** 2
    
    # Convertir en numpy
    X_cpu = cp.asnumpy(X_valid)
    Y_cpu = cp.asnumpy(Y_valid)
    S_cpu = cp.asnumpy(S_valid)
    
    return X_cpu, Y_cpu, S_cpu, (w, h)

def halftone_gpu_scatter_fullres(image_path, cell=4, max_radius=None, cells_per_pixel=1):
    """
    Version scatter plot pour visualisation, résolution complète
    Utilise une approche par grille sans map_coordinates
    """
    
    if max_radius is None:
        max_radius = cell / 2.2
    
    # Charger l'image
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    h, w = img_array.shape
    
    # Transférer vers GPU
    img_gpu = cp.asarray(img_array, dtype=cp.float32)
    
    # Ajuster la taille de cellule effective
    effective_cell = max(1, cell // cells_per_pixel)
    
    # Calculer les dimensions de la grille
    grid_h = h // effective_cell
    grid_w = w // effective_cell
    
    # Redimensionner pour être divisible par effective_cell
    cropped_h = grid_h * effective_cell
    cropped_w = grid_w * effective_cell
    img_cropped = img_gpu[:cropped_h, :cropped_w]
    
    # Reshaper pour créer les blocs
    blocks = img_cropped.reshape(grid_h, effective_cell, grid_w, effective_cell)
    mean_intensities = cp.mean(blocks, axis=(1, 3)) / 255.0
    
    # Calculer les rayons
    radii = (1 - mean_intensities) * max_radius
    
    # Listes pour stocker tous les points
    all_X = []
    all_Y = []
    all_S = []
    
    # Convertir radii en numpy pour l'itération
    radii_cpu = cp.asnumpy(radii)
    
    # Générer les points pour chaque cellule de la grille
    for i in range(grid_h):
        for j in range(grid_w):
            if radii_cpu[i, j] > 0.1:
                # Position du centre de la cellule
                base_y = i * effective_cell + effective_cell // 2
                base_x = j * effective_cell + effective_cell // 2
                
                # Créer des sous-points selon cells_per_pixel
                if cells_per_pixel == 1:
                    # Un seul point au centre
                    all_X.append(base_x)
                    all_Y.append(base_y)
                    all_S.append((radii_cpu[i, j] * 2) ** 2)
                else:
                    # Plusieurs points répartis dans la cellule
                    spacing = effective_cell // cells_per_pixel
                    for dy in range(cells_per_pixel):
                        for dx in range(cells_per_pixel):
                            offset_y = (dy - (cells_per_pixel-1)/2) * spacing
                            offset_x = (dx - (cells_per_pixel-1)/2) * spacing
                            
                            point_x = base_x + offset_x
                            point_y = base_y + offset_y
                            
                            # Vérifier les limites
                            if 0 <= point_x < w and 0 <= point_y < h:
                                all_X.append(point_x)
                                all_Y.append(point_y)
                                # Taille réduite pour les points multiples
                                point_size = ((radii_cpu[i, j] / cells_per_pixel) * 2) ** 2
                                all_S.append(point_size)
    
    return np.array(all_X), np.array(all_Y), np.array(all_S), (w, h)

def create_halftone_preserved(image_path, cell_size=4, output_path=None, max_radius=None, cells_per_pixel=1):
    """
    Fonction principale compatible avec votre code existant
    """
    if output_path is None:
        output_path = f"halftone_cell_{cell_size}.png"
    
    if max_radius is None:
        max_radius = cell_size / 2.2
    
    # Utiliser la méthode grille pour éviter l'erreur map_coordinates
    X, Y, S, (w, h) = halftone_gpu_scatter_fullres(image_path, cell_size, max_radius, cells_per_pixel)
    
    # Affichage
    plt.figure(figsize=(12, 12))
    plt.scatter(X, -Y, c="black", s=S, marker="o", alpha=0.8)
    plt.xlim(0, w)
    plt.ylim(-h, 0)
    plt.axis("equal")
    plt.axis("off")
    plt.title(f'Halftone - Cell: {cell_size}, Points: {len(X)}')
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor='white')
    plt.show()
    
    print(f"Halftone sauvegardé: {output_path}")
    print(f"Nombre de points: {len(X)}")
    print(f"Résolution de référence: {w}x{h}")
    
    return X, Y, S

def create_halftone_gpu_fullres(image_path="test5.jpg", cell=4, output_path="halftone_fullres.png", 
                                radius=0.1, cells_per_pixel=1, method="bitmap"):
    """
    Crée un halftone en résolution complète
    
    Parameters:
    - method: "bitmap" pour image bitmap, "scatter" pour scatter plot
    - cells_per_pixel: 1 pour halftone classique, >1 pour plus de détails
    """
    
    if method == "bitmap":
        # Méthode bitmap - produit une vraie image
        halftone_img = halftone_gpu_optimized_fullres(image_path, cell, radius, cells_per_pixel)
        
        # Sauvegarder l'image
        Image.fromarray(halftone_img.astype(np.uint8)).save(output_path)
        
        # Affichage
        plt.figure(figsize=(10, 10))
        plt.imshow(halftone_img, cmap='gray')
        plt.axis('off')
        plt.title(f'Halftone - Cell: {cell}, Cells/pixel: {cells_per_pixel}')
        plt.tight_layout()
        plt.show()
        
        print(f"Halftone bitmap sauvegardé: {output_path}")
        print(f"Résolution: {halftone_img.shape}")
        
    else:  # method == "scatter"
        # Méthode scatter plot
        X, Y, S, (w, h) = halftone_gpu_scatter_fullres(image_path, cell, radius, cells_per_pixel)
        
        # Affichage
        plt.figure(figsize=(12, 12))
        plt.scatter(X, -Y, c="black", s=S, marker="o")
        plt.xlim(0, w)
        plt.ylim(-h, 0)
        plt.axis("equal")
        plt.axis("off")
        plt.title(f'Halftone Scatter - Cell: {cell}, Cells/pixel: {cells_per_pixel}')
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.show()
        
        print(f"Halftone scatter sauvegardé: {output_path}")
        print(f"Nombre de points: {len(X)}")
        print(f"Résolution de référence: {w}x{h}")

# Exemples d'utilisation
if __name__ == "__main__":

    # Version scatter plot pour comparaison
    create_halftone_gpu_fullres("flou_surface_interactif.jpg", 
                               cell=15, 
                               radius=2.0,
                               cells_per_pixel=4,
                               method="scatter",
                               output_path="halftone_scatter_fullres.png")