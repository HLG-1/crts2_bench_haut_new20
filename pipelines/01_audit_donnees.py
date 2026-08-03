import sys
sys.path.append(".")
from core.geo_io import auditer_zone

ZONES = {
    "zone1": ("data/zone1/01_vec.shp", "data/zone1/01.tif"),
}

if __name__ == "__main__":
    resultats = {}
    for nom, (shp, tif) in ZONES.items():
        resultats[nom] = auditer_zone(nom, shp, tif)

    if not all(ok for _, ok in resultats.values()):
        print("\n  Au moins une zone n'a pas de chevauchement vecteur/raster — corriger avant de continuer.")
    else:
        print("\n Toutes les zones sont alignées, vous pouvez lancer 02_preparation_data.py")