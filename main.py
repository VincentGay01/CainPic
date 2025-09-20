
import argparse
from halftone import create_halftone_gpu 
from damier import flou_interactif


parser = argparse.ArgumentParser(description="yabadabadou")
parser.add_argument("image", help="Chemin de l'image à traiter")
parser.add_argument("--cell", type=int, default=3, help="Taille de la grille pour le halftone")
parser.add_argument("--type",default='motion' ,help="type de flou ")
parser.add_argument("--strength", type=int, default=20, help="puissance du flou ")

args = parser.parse_args()
print("Image :", args.image)
print("type de flou :",args.type)
print("puissance du flou :",args.strength)
print("Taille de cellule :", args.cell)


flou_interactif(args.image, blur_type=args.type, blur_strength=args.strength)

create_halftone_gpu(f"flou_{args.type}_interactif.jpg", cell=args.cell)
