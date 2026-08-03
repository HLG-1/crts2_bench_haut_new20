import sys
sys.path.append(".")
from core.geo_io import auditer_zone
from core.rasterization import nettoyer_gdf
from core.tiling import decouper_en_patches
from core.splits import generer_splits


ZONES = {
    "zone1": ("data/zone1/01_vec.shp", "data/zone1/01.tif"),
}

if __name__ == "__main__":
    manifests = {}
    for zone_name, (shp, tif) in ZONES.items():
        gdf_reproj, ok = auditer_zone(zone_name, shp, tif)
        if not ok:
            raise RuntimeError(f"{zone_name}: pas de chevauchement, arrêt.")

        gdf_clean = nettoyer_gdf(gdf_reproj)
        print(f"{zone_name}: {len(gdf_clean)} polygones valides (sur {len(gdf_reproj)})")

        out_dir = f"data/patches/{zone_name}"
        manifests[zone_name] = decouper_en_patches(tif, gdf_clean, out_dir, zone_name)
        print(f"{zone_name}: {len(manifests[zone_name])} patches générés")

    # train_val = manifests["zone1"] + manifests["zone2"]
    # generer_splits(train_val, manifests["zone3"], "data/splits")