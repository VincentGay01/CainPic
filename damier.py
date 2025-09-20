import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import cv2
from scipy.ndimage import gaussian_filter

class FlouteurZone:
    def __init__(self, image_path):
        """Initialise avec une image couleur"""
        self.image = Image.open(image_path).convert("RGB")
        self.img_array = np.array(self.image)
        self.height, self.width = self.img_array.shape[:2]
        print(f"Image chargée: {self.width}x{self.height}")
    
    def gaussian_blur(self, image_array, sigma):
        """Applique un flou gaussien"""
        if len(image_array.shape) == 3:  # Image couleur
            blurred = np.zeros_like(image_array)
            for i in range(3):  # RGB
                blurred[:, :, i] = gaussian_filter(image_array[:, :, i], sigma=sigma)
            return blurred
        else:  # Image en niveaux de gris
            return gaussian_filter(image_array, sigma=sigma)
    
    def motion_blur(self, image_array, kernel_size, angle=0):
        """Applique un flou de mouvement"""
        # Créer un noyau de flou directionnel
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Ligne centrale pour l'angle 0
        if angle == 0:
            kernel[kernel_size//2, :] = 1
        else:
            # Rotation du noyau pour autres angles
            center = kernel_size // 2
            for i in range(kernel_size):
                x = int(center + (i - center) * np.cos(np.radians(angle)))
                y = int(center + (i - center) * np.sin(np.radians(angle)))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1
        
        kernel = kernel / np.sum(kernel)  # Normaliser
        
        # Appliquer le flou
        if len(image_array.shape) == 3:
            blurred = np.zeros_like(image_array)
            for i in range(3):
                blurred[:, :, i] = cv2.filter2D(image_array[:, :, i], -1, kernel)
            return blurred
        else:
            return cv2.filter2D(image_array, -1, kernel)
    
    def box_blur(self, image_array, kernel_size):
        """Applique un flou en boîte (moyennage)"""
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        if len(image_array.shape) == 3:
            blurred = np.zeros_like(image_array)
            for i in range(3):
                blurred[:, :, i] = cv2.filter2D(image_array[:, :, i], -1, kernel)
            return blurred
        else:
            return cv2.filter2D(image_array, -1, kernel)
    
    def apply_blur_rectangle(self, x1, y1, x2, y2, blur_type='gaussian', 
                           blur_strength=5, fade_edges=True, fade_width=20):
        """Applique un flou sur une zone rectangulaire"""
        # S'assurer que les coordonnées sont dans l'image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            print("Zone invalide!")
            return self.img_array.copy()
        
        result = self.img_array.copy()
        zone = result[y1:y2, x1:x2].copy()
        
        # Appliquer le type de flou choisi
        if blur_type == 'gaussian':
            blurred_zone = self.gaussian_blur(zone, sigma=blur_strength)
        elif blur_type == 'motion':
            blurred_zone = self.motion_blur(zone, kernel_size=blur_strength*2+1, angle=0)
        elif blur_type == 'box':
            blurred_zone = self.box_blur(zone, kernel_size=blur_strength*2+1)
        elif blur_type == 'radial':
            # Flou radial (effet zoom)
            blurred_zone = self.radial_blur(zone, blur_strength)
        else:
            blurred_zone = self.gaussian_blur(zone, sigma=blur_strength)
        
        if fade_edges and fade_width > 0:
            # Créer un masque de fondu pour les bords
            mask = self.create_fade_mask(x2-x1, y2-y1, fade_width)
            
            # Mélanger avec le masque
            for i in range(3):
                result[y1:y2, x1:x2, i] = (
                    zone[:, :, i] * (1 - mask) + 
                    blurred_zone[:, :, i] * mask
                ).astype(np.uint8)
        else:
            result[y1:y2, x1:x2] = blurred_zone.astype(np.uint8)
        
        return result
    
    
    def radial_blur(self, image_array, strength):
        """Flou radial (effet zoom/rotation)"""
        h, w = image_array.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Créer une grille de coordonnées
        y, x = np.ogrid[:h, :w]
        
        # Calculer la distance au centre
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Normaliser la distance
        normalized_distance = distance / max_distance
        
        # Appliquer un flou variable selon la distance
        result = image_array.copy().astype(float)
        
        for i in range(strength):
            # Flou progressif plus fort vers les bords
            sigma = normalized_distance * (i + 1) * 0.5
            if len(image_array.shape) == 3:
                for c in range(3):
                    blurred = gaussian_filter(image_array[:, :, c], sigma=1)
                    # Mélanger selon la distance
                    blend_factor = normalized_distance * 0.1
                    result[:, :, c] = result[:, :, c] * (1 - blend_factor) + blurred * blend_factor
            
        return result.astype(np.uint8)
    
    def create_fade_mask(self, width, height, fade_width):
        """Crée un masque de fondu pour les bords"""
        mask = np.ones((height, width))
        
        # Créer le fondu sur les bords
        for i in range(fade_width):
            alpha = i / fade_width
            
            # Bords horizontaux
            if i < height:
                mask[i, :] = alpha
                mask[height-1-i, :] = alpha
            
            # Bords verticaux  
            if i < width:
                mask[:, i] = np.minimum(mask[:, i], alpha)
                mask[:, width-1-i] = np.minimum(mask[:, width-1-i], alpha)
        
        return mask
    
    def interactive_selection(self):
        """Sélection interactive de zone avec matplotlib"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.img_array)
        ax.set_title("Cliquez pour sélectionner les coins d'un rectangle à flouter\n(2 clics: coin haut-gauche puis bas-droite)")
        
        coords = []
        
        def onclick(event):
            if event.inaxes != ax:
                return
            
            coords.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
            
            if len(coords) == 2:
                # Dessiner le rectangle
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                rect = plt.Rectangle((min(x1, x2), min(y1, y2)), 
                                   abs(x2-x1), abs(y2-y1), 
                                   fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                plt.title("Zone sélectionnée! Fermez la fenêtre pour continuer.")
            
            fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        
        if len(coords) == 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        else:
            print("Sélection annulée")
            return None
    
    def show_results(self, results, titles):
        """Affiche plusieurs résultats côte à côte"""
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 8))
        if n == 1:
            axes = [axes]
        
        for i, (result, title) in enumerate(zip(results, titles)):
            axes[i].imshow(result)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Fonctions d'utilisation simple
def flou_rectangle_simple(image_path, x1, y1, x2, y2, blur_type='gaussian', 
                         blur_strength=5, output_path="flou_result.jpg"):
    """Version simple pour rectangle"""
    flouteur = FlouteurZone(image_path)
    result = flouteur.apply_blur_rectangle(x1, y1, x2, y2, blur_type, blur_strength)
    
    # Sauvegarder
    Image.fromarray(result).save(output_path)
    
    # Afficher
    flouteur.show_results([flouteur.img_array, result], ["Original", f"Flou {blur_type}"])
    
    return result

def flou_interactif(image_path, blur_type='gaussian', blur_strength=5):
    """Version interactive"""
    flouteur = FlouteurZone(image_path)
    
    # Sélection interactive
    selection = flouteur.interactive_selection()
    if selection:
        x1, y1, x2, y2 = selection
        result = flouteur.apply_blur_rectangle(x1, y1, x2, y2, blur_type, blur_strength)
        
        flouteur.show_results([flouteur.img_array, result], ["Original", f"Flou {blur_type}"])
        
        # Sauvegarder
        Image.fromarray(result).save(f"flou_{blur_type}_interactif.jpg")
        print(f"Résultat sauvegardé: flou_{blur_type}_interactif.jpg")
        
        return result
    return None

# Exemples d'utilisation
if __name__ == "__main__":
    # Remplacez par le chemin de votre image
    IMAGE_PATH = "test5.jpg"
    print("\n Exemple 4: Sélection interactive")
    print("Cliquez sur l'image pour sélectionner une zone à flouter...")
    flou_interactif(IMAGE_PATH, blur_type='motion', blur_strength=20)
        