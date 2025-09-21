
import argparse
from halftone import create_halftone_gpu 
from damier import flou_interactif
from contour import enhance_contours_simple

parser = argparse.ArgumentParser(description="yabadabadou")
parser.add_argument("image", help="Chemin de l'image à traiter")
parser.add_argument("--cell", type=int, default=3, help="Taille de la grille pour le halftone")
parser.add_argument("--radius",type=float, default=0.1 ,help="radius des points")
parser.add_argument("--type",default='motion' ,help="type de flou ")
parser.add_argument("--strength", type=int, default=20, help="puissance du flou ")
parser.add_argument("--choix", type=int, default=0, help="avec contrast")
args = parser.parse_args()
print("Image :", args.image)
print("type de flou :",args.type)
print("puissance du flou :",args.strength)
print("Taille de cellule :", args.cell)
print("avec ou sans contrast :", args.choix)



if(args.choix==0):
    enhanced = enhance_contours_simple(
                args.image, 
                threshold=30,   # Seuil de détection
                strength=1.8,   # Force du renforcement 
                method='overlay', # Méthode d'amélioration
                output_path="contours_sobel.jpg"
            )

    flou_interactif("contours_sobel.jpg", blur_type=args.type, blur_strength=args.strength)

    create_halftone_gpu(f"flou_{args.type}_interactif.jpg", cell=args.cell)
else:
    flou_interactif(args.image, blur_type=args.type, blur_strength=args.strength)

    create_halftone_gpu(f"flou_{args.type}_interactif.jpg", cell=args.cell)
    