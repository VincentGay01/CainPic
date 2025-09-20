import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel

class SobelContourEnhancer:
    def __init__(self, image_path):
        """Initialise avec une image"""
        self.image = Image.open(image_path).convert("L")
        self.img_array = np.array(self.image, dtype=np.float32)
        self.height, self.width = self.img_array.shape
        print(f"Image chargée: {self.width}x{self.height}")
    
    def detect_contours_sobel(self, threshold=30):
        """Détecte les contours avec Sobel"""
        # Gradients Sobel
        grad_x = sobel(self.img_array, axis=1)  # Gradient horizontal
        grad_y = sobel(self.img_array, axis=0)  # Gradient vertical
        
        # Magnitude du gradient (force des contours)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Créer un masque des contours
        contours_mask = (gradient_magnitude > threshold).astype(np.float32)
        
        return gradient_magnitude, contours_mask
    
    def enhance_contours(self, threshold=30, strength=1.5, method='darken'):
        """Met en avant les contours avec différentes méthodes"""
        gradient_magnitude, contours_mask = self.detect_contours_sobel(threshold)
        
        # Copie de l'image originale
        enhanced = self.img_array.copy()
        
        if method == 'darken':
            # Assombrir les contours
            enhanced = enhanced - (contours_mask * strength * 50)
            enhanced = np.clip(enhanced, 0, 255)
        
        elif method == 'lighten':
            # Éclaircir les contours
            enhanced = enhanced + (contours_mask * strength * 50)
            enhanced = np.clip(enhanced, 0, 255)
        
        elif method == 'contrast':
            # Augmenter le contraste sur les contours
            # Zones de contours : plus sombres si déjà sombres, plus claires si déjà claires
            threshold_img = enhanced > 127  # Seuil milieu
            
            # Assombrir les zones sombres avec contours
            dark_contours = contours_mask * (~threshold_img)
            enhanced = enhanced - (dark_contours * strength * 40)
            
            # Éclaircir les zones claires avec contours
            light_contours = contours_mask * threshold_img
            enhanced = enhanced + (light_contours * strength * 40)
            
            enhanced = np.clip(enhanced, 0, 255)
        
        elif method == 'invert':
            # Inverser seulement les contours
            inverted_contours = 255 - enhanced
            enhanced = enhanced * (1 - contours_mask) + inverted_contours * contours_mask
            enhanced = np.clip(enhanced, 0, 255)
        
        elif method == 'overlay':
            # Superposer les contours en noir
            enhanced = enhanced * (1 - contours_mask * strength * 0.8)
            enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced, contours_mask * 255, gradient_magnitude
    
    def adaptive_enhance(self, base_threshold=20, fine_threshold=40, 
                        base_strength=1.0, fine_strength=2.0):
        """Amélioration adaptative avec deux niveaux de contours"""
        # Contours grossiers (structures principales)
        grad_mag_coarse, contours_coarse = self.detect_contours_sobel(base_threshold)
        
        # Contours fins (détails)
        grad_mag_fine, contours_fine = self.detect_contours_sobel(fine_threshold)
        
        # Amélioration progressive
        enhanced = self.img_array.copy()
        
        # Renforcer les contours grossiers modérément
        enhanced = enhanced - (contours_coarse * base_strength * 30)
        
        # Renforcer les contours fins plus fortement
        enhanced = enhanced - (contours_fine * fine_strength * 60)
        
        enhanced = np.clip(enhanced, 0, 255)
        
        # Masque combiné pour affichage
        combined_contours = np.maximum(contours_coarse, contours_fine) * 255
        
        return enhanced, combined_contours, grad_mag_coarse + grad_mag_fine
    
    def show_results(self, enhanced, contours, gradient=None, method_name="Sobel"):
        """Affiche les résultats"""
        if gradient is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Image originale
            axes[0, 0].imshow(self.img_array, cmap='gray')
            axes[0, 0].set_title("Image Originale")
            axes[0, 0].axis('off')
            
            # Gradient magnitude
            axes[0, 1].imshow(gradient, cmap='gray')
            axes[0, 1].set_title("Gradient Magnitude (Sobel)")
            axes[0, 1].axis('off')
            
            # Contours détectés
            axes[1, 0].imshow(contours, cmap='gray')
            axes[1, 0].set_title("Contours Détectés")
            axes[1, 0].axis('off')
            
            # Image avec contours renforcés
            axes[1, 1].imshow(enhanced, cmap='gray')
            axes[1, 1].set_title(f"Contours Renforcés ({method_name})")
            axes[1, 1].axis('off')
        
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Image originale
            axes[0].imshow(self.img_array, cmap='gray')
            axes[0].set_title("Image Originale")
            axes[0].axis('off')
            
            # Contours détectés
            axes[1].imshow(contours, cmap='gray')
            axes[1].set_title("Contours Détectés")
            axes[1].axis('off')
            
            # Image avec contours renforcés
            axes[2].imshow(enhanced, cmap='gray')
            axes[2].set_title(f"Contours Renforcés ({method_name})")
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_result(self, enhanced_image, output_path="contours_enhanced.jpg"):
        """Sauvegarde le résultat"""
        enhanced_pil = Image.fromarray(enhanced_image.astype(np.uint8))
        enhanced_pil.save(output_path)
        print(f"💾 Image sauvegardée: {output_path}")

def enhance_contours_simple(image_path, threshold=30, strength=1.5, 
                           method='darken', output_path="contours_enhanced.jpg"):
    """Fonction simple pour renforcer les contours"""
    enhancer = SobelContourEnhancer(image_path)
    
    # Renforcer les contours
    enhanced, contours, gradient = enhancer.enhance_contours(
        threshold=threshold, strength=strength, method=method
    )
    
    # Afficher les résultats
    #enhancer.show_results(enhanced, contours, gradient, method_name=method)
    
    # Sauvegarder
    enhancer.save_result(enhanced, output_path)
    
    print(f"  Paramètres: threshold={threshold}, strength={strength}, method={method}")
    
    return enhanced





