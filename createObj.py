from PIL import Image
import numpy as np
import trimesh

# --- 1. Charger l'image et récupérer l'alpha ---
img = Image.open("policeMain.png").convert("RGBA")
alpha = np.array(img)[:, :, 3]  # canal alpha

# --- 2. Créer un masque binaire ---
mask = alpha > 0

# --- 3. Générer des points 3D ---
verts = []
faces = []
h = 1.0  # hauteur d’extrusion

rows, cols = mask.shape
for y in range(rows):
    for x in range(cols):
        if mask[y, x]:
            # Chaque pixel opaque = petit carré extrudé
            i = len(verts)
            verts.extend([
                [x, y, 0],
                [x+1, y, 0],
                [x+1, y+1, 0],
                [x, y+1, 0],
                [x, y, h],
                [x+1, y, h],
                [x+1, y+1, h],
                [x, y+1, h],
            ])
            faces.extend([
                [i, i+1, i+2], [i, i+2, i+3],       # bas
                [i+4, i+5, i+6], [i+4, i+6, i+7],   # haut
                [i, i+1, i+5], [i, i+5, i+4],       # côtés
                [i+1, i+2, i+6], [i+1, i+6, i+5],
                [i+2, i+3, i+7], [i+2, i+7, i+6],
                [i+3, i, i+4], [i+3, i+4, i+7],
            ])

# --- 4. Construire le mesh et exporter ---
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export("surface.obj")
