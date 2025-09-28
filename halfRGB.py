import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cupyx.scipy import ndimage

def halftone_gpu_rgb(image_path, cell=4, max_radius=None, channels='RGB', cells_per_pixel=1, 
                    channel_weights=None):
    """
    Crée un halftone coloré avec sélection de canaux RGB
    
    Parameters:
    - image_path: chemin vers l'image
    - cell: taille de la cellule
    - max_radius: rayon maximum des points
    - channels: 'R', 'G', 'B', 'RG', 'RB', 'GB', 'RGB', ou 'CMYK'
    - cells_per_pixel: nombre de cellules par pixel
    - channel_weights: dict pour ajuster l'intensité {'R': 1.0, 'G': 1.0, 'B': 0.7, 'C': 1.0, 'M': 1.0, 'Y': 1.0, 'K': 1.0}
    """
    
    # Poids par défaut - le bleu réduit pour compenser la sur-représentation visuelle
    if channel_weights is None:
        channel_weights = {
            'R': 1.0,    # Rouge normal
            'G': 1.0,    # Vert normal  
            'B': 0.7,    # Bleu réduit (plus réaliste visuellement)
            'C': 1.0,    # Cyan normal
            'M': 1.0,    # Magenta normal
            'Y': 1.0,    # Jaune normal
            'K': 1.0     # Noir normal
        }
    
    if max_radius is None:
        max_radius = cell / 2.2
    
    # Charger l'image en couleur
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w, c = img_array.shape
    
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
    
    # Séparer les canaux RGB
    r_channel = img_cropped[:, :, 0]
    g_channel = img_cropped[:, :, 1]
    b_channel = img_cropped[:, :, 2]
    
    # Fonction pour traiter un canal avec pondération
    def process_channel(channel_data, channel_name, color, weight=1.0):
        blocks = channel_data.reshape(grid_h, effective_cell, grid_w, effective_cell)
        mean_intensities = cp.mean(blocks, axis=(1, 3)) / 255.0
        
        # Appliquer la pondération du canal
        weighted_intensities = mean_intensities * weight
        weighted_intensities = cp.clip(weighted_intensities, 0.0, 1.0)  # S'assurer de rester dans [0,1]
        
        radii = (1 - weighted_intensities) * max_radius
        
        points_x = []
        points_y = []
        points_s = []
        points_c = []
        
        radii_cpu = cp.asnumpy(radii)
        
        for i in range(grid_h):
            for j in range(grid_w):
                if radii_cpu[i, j] > 0.1:
                    base_y = i * effective_cell + effective_cell // 2
                    base_x = j * effective_cell + effective_cell // 2
                    
                    if cells_per_pixel == 1:
                        points_x.append(base_x)
                        points_y.append(base_y)
                        points_s.append((radii_cpu[i, j] * 2) ** 2)
                        points_c.append(color)
                    else:
                        spacing = effective_cell // cells_per_pixel
                        for dy in range(cells_per_pixel):
                            for dx in range(cells_per_pixel):
                                offset_y = (dy - (cells_per_pixel-1)/2) * spacing
                                offset_x = (dx - (cells_per_pixel-1)/2) * spacing
                                
                                point_x = base_x + offset_x
                                point_y = base_y + offset_y
                                
                                if 0 <= point_x < w and 0 <= point_y < h:
                                    points_x.append(point_x)
                                    points_y.append(point_y)
                                    point_size = ((radii_cpu[i, j] / cells_per_pixel) * 2) ** 2
                                    points_s.append(point_size)
                                    points_c.append(color)
        
        return points_x, points_y, points_s, points_c
    
    # Traiter les canaux sélectionnés
    all_X = []
    all_Y = []
    all_S = []
    all_C = []
    
    if channels.upper() == 'CMYK':
        # Mode CMYK - conversion des canaux RGB en CMY + K
        # Calculer K (noir) - valeur minimum des trois canaux
        k_channel = cp.minimum(cp.minimum(r_channel, g_channel), b_channel)
        
        # Calculer CMY en évitant la division par zéro
        # Créer un masque pour éviter la division par zéro
        safe_denominator = cp.maximum(255 - k_channel, 1e-8)  # Évite division par zéro
        
        # Calculer les canaux CMY
        c_channel = cp.where(k_channel < 254, 
                            cp.clip((255 - r_channel - k_channel) / safe_denominator * 255, 0, 255), 
                            0)
        m_channel = cp.where(k_channel < 254, 
                            cp.clip((255 - g_channel - k_channel) / safe_denominator * 255, 0, 255), 
                            0)
        y_channel = cp.where(k_channel < 254, 
                            cp.clip((255 - b_channel - k_channel) / safe_denominator * 255, 0, 255), 
                            0)
        
        # Traiter chaque canal CMYK avec décalage pour éviter superposition
        offset = max(1, effective_cell // 8)  # Petit décalage pour séparer les couleurs
        
        # Cyan (décalage haut-gauche)
        x, y, s, c = process_channel(c_channel, 'C', 'cyan', channel_weights.get('C', 1.0))
        all_X.extend([xi - offset for xi in x])
        all_Y.extend([yi - offset for yi in y])
        all_S.extend(s)
        all_C.extend(c)
        
        # Magenta (décalage haut-droite)  
        x, y, s, c = process_channel(m_channel, 'M', 'magenta', channel_weights.get('M', 1.0))
        all_X.extend([xi + offset for xi in x])
        all_Y.extend([yi - offset for yi in y])
        all_S.extend(s)
        all_C.extend(c)
        
        # Yellow (décalage bas-gauche)
        x, y, s, c = process_channel(y_channel, 'Y', 'yellow', channel_weights.get('Y', 1.0))
        all_X.extend([xi - offset for xi in x])
        all_Y.extend([yi + offset for yi in y])
        all_S.extend(s)
        all_C.extend(c)
        
        # Black (décalage bas-droite)
        x, y, s, c = process_channel(k_channel, 'K', 'black', channel_weights.get('K', 1.0))
        all_X.extend([xi + offset for xi in x])
        all_Y.extend([yi + offset for yi in y])
        all_S.extend(s)
        all_C.extend(c)
        
    else:
        # Mode RGB classique
        offset = max(1, effective_cell // 6)  # Décalage pour séparer les canaux
        
        if 'R' in channels.upper():
            x, y, s, c = process_channel(r_channel, 'R', 'red', channel_weights.get('R', 1.0))
            all_X.extend([xi - offset for xi in x])
            all_Y.extend([yi - offset for yi in y])
            all_S.extend(s)
            all_C.extend(c)
        
        if 'G' in channels.upper():
            x, y, s, c = process_channel(g_channel, 'G', 'green', channel_weights.get('G', 1.0))
            all_X.extend(x)  # Pas de décalage pour le vert (centre)
            all_Y.extend(y)
            all_S.extend(s)
            all_C.extend(c)
        
        if 'B' in channels.upper():
            x, y, s, c = process_channel(b_channel, 'B', 'blue', channel_weights.get('B', 0.7))  # Bleu réduit par défaut
            all_X.extend([xi + offset for xi in x])
            all_Y.extend([yi + offset for yi in y])
            all_S.extend(s)
            all_C.extend(c)
    
    return np.array(all_X), np.array(all_Y), np.array(all_S), all_C, (w, h)

def halftone_gpu_rgb_bitmap(image_path, cell=4, max_radius=None, channels='RGB', cells_per_pixel=1, 
                           channel_weights=None):
    """
    Version bitmap qui génère une vraie image colorée
    """
    
    if max_radius is None:
        max_radius = cell / 2.2
    
    # Charger l'image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w, c = img_array.shape
    
    # Créer image de sortie (fond blanc)
    output_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Obtenir les points colorés
    X, Y, S, colors, _ = halftone_gpu_rgb(image_path, cell, max_radius, channels, cells_per_pixel, channel_weights)
    
    # Dessiner les points
    color_map = {
        'red': [255, 0, 0],
        'green': [0, 255, 0], 
        'blue': [0, 0, 255],
        'cyan': [0, 255, 255],
        'magenta': [255, 0, 255],
        'yellow': [255, 255, 0],
        'black': [0, 0, 0]
    }
    
    for x, y, size, color in zip(X, Y, S, colors):
        radius = int(np.sqrt(size) / 2)
        x, y = int(x), int(y)
        
        # Dessiner un cercle rempli
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    py, px = y + dy, x + dx
                    if 0 <= py < h and 0 <= px < w:
                        # Mélange additif pour les couleurs qui se superposent
                        current_color = output_img[py, px]
                        new_color = color_map[color]
                        # Utiliser le mode de mélange "multiply" pour un effet plus réaliste
                        blended = ((current_color.astype(float) / 255) * (np.array(new_color) / 255) * 255).astype(np.uint8)
                        output_img[py, px] = blended
    
    return output_img

def create_halftone_rgb(image_path, cell=4, output_path=None, max_radius=None, 
                       channels='RGB', cells_per_pixel=1, method='scatter', alpha=0.7, 
                       channel_weights=None):
    """
    Fonction principale pour créer des halftones colorés
    
    Parameters:
    - channels: 'R', 'G', 'B', 'RG', 'RB', 'GB', 'RGB', ou 'CMYK'
    - method: 'scatter' ou 'bitmap'
    - alpha: transparence des points (pour method='scatter')
    - channel_weights: dict pour contrôler l'intensité des canaux
                      Ex: {'R': 1.0, 'G': 1.0, 'B': 0.5} pour réduire le bleu
    """
    
    if output_path is None:
        output_path = f"halftone_{channels}_cell_{cell}.png"
    
    if max_radius is None:
        max_radius = cell / 2.2
    
    if method == 'bitmap':
        # Méthode bitmap
        halftone_img = halftone_gpu_rgb_bitmap(image_path, cell, max_radius, channels, cells_per_pixel, channel_weights)
        
        # Sauvegarder
        Image.fromarray(halftone_img).save(output_path)
        
        # Afficher
        plt.figure(figsize=(12, 12))
        plt.imshow(halftone_img)
        plt.axis('off')
        plt.title(f'Halftone {channels} - Cell: {cell}, Cells/pixel: {cells_per_pixel}')
        plt.tight_layout()
        plt.show()
        
        print(f"Halftone bitmap {channels} sauvegardé: {output_path}")
        
    else:  # method == 'scatter'
        # Méthode scatter plot
        X, Y, S, colors, (w, h) = halftone_gpu_rgb(image_path, cell, max_radius, channels, cells_per_pixel, channel_weights)
        
        # Affichage
        plt.figure(figsize=(12, 12), facecolor='white')
        plt.scatter(X, -Y, c=colors, s=S, marker="o", alpha=alpha, edgecolors='none')
        plt.xlim(0, w)
        plt.ylim(-h, 0)
        plt.axis("equal")
        plt.axis("off")
        plt.title(f'Halftone {channels} - Cell: {cell}, Points: {len(X)}')
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor='white')
        plt.show()
        
        print(f"Halftone scatter {channels} sauvegardé: {output_path}")
        print(f"Nombre de points: {len(X)}")
    
    return output_path

def create_halftone_preserved(image_path, cell_size=4, output_path=None, max_radius=None, 
                             channels='RGB', cells_per_pixel=1, method='scatter', alpha=0.7, 
                             channel_weights=None):
    """
    Fonction compatible avec votre code existant - maintenant avec couleurs et contrôle d'intensité !
    """
    return create_halftone_rgb(image_path, cell_size, output_path, max_radius, 
                              channels, cells_per_pixel, method, alpha, channel_weights)

# Exemples d'utilisation
if __name__ == "__main__":
    image_path = "D:/project/CainPic/test/test5.jpg"
    
    # Halftone RGB avec bleu réduit (par défaut)
    #create_halftone_rgb(image_path, cell=6, channels='RGB', method='scatter', alpha=0.8)
    
    # Contrôle personnalisé des canaux - bleu encore plus réduit
    #custom_weights = {'R': 1.5, 'G': 1.5, 'B': 0.2}
    #create_halftone_rgb(image_path, cell=10, channels='RGB', method='scatter', 
    #                   alpha=0.4, channel_weights=custom_weights)
    
    # Rouge et vert uniquement (pas de bleu du tout)
    #create_halftone_rgb(image_path, cell=6, channels='RG', method='scatter', alpha=0.5)
    
    # Mode CMYK avec cyan et magenta réduits
    cmyk_weights = {'C': 0.8, 'M': 0.8, 'Y': 1.0, 'K': 1.2}
    create_halftone_rgb(image_path, cell=8, channels='CMYK', method='scatter', 
                       alpha=0.6, channel_weights=cmyk_weights)
    
    # Effet vintage - rouge dominant
    #vintage_weights = {'R': 1.3, 'G': 0.8, 'B': 0.5}
    #create_halftone_rgb(image_path, cell=10, channels='RGB', method='scatter',
    #                   alpha=0.2, channel_weights=vintage_weights)