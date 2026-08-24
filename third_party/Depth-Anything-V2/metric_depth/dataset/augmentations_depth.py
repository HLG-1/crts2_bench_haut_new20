"""
augmentations_depth.py

Pipeline d'augmentation pour l'entraînement DAV2 sur BatimentsMarocDataset.

Contrairement à un pipeline de segmentation classique, ici on transforme
TROIS éléments de façon cohérente :
    - image RGB          (subit géométrie + photométrie)
    - depth map           (subit UNIQUEMENT la géométrie, jamais la photométrie)
    - valid_mask           (subit UNIQUEMENT la géométrie, jamais la photométrie)

Albumentations gère ça nativement via `additional_targets`, qui applique
automatiquement les transformations géométriques (crop, flip, rotate, resize)
à tous les targets déclarés, mais restreint les transformations photométriques
(couleur, bruit, flou) au seul target "image".

Points d'attention specifiques depth (vs segmentation classique) :
    1. CoarseDropout doit mettre valid_mask=0 sur les trous, PAS depth=0,
       pour ne jamais injecter de fausses valeurs de profondeur nulle
       dans la loss (SiLogLoss).
    2. Les transformations photométriques ne doivent JAMAIS toucher depth
       ou valid_mask (Albumentations le garantit via mask targets).
    3. RandomResizedCrop change l'échelle apparente -> cohérent car la depth
       est en mètres absolus (pas relative à l'image), donc pas de
       recalibration nécessaire après crop/resize.
"""
from __future__ import annotations

import albumentations as A
import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _random_resized_crop(image_size: tuple[int, int], scale, ratio) -> A.BasicTransform:
    """
    Compatibilité entre versions d'Albumentations :
    - anciennes versions : RandomResizedCrop(height=, width=, ...)
    - versions récentes  : RandomResizedCrop(size=(h, w), ...)

    IMPORTANT : p=1.0 (fixe, non paramétrable) volontairement.
    Ce transform garantit la taille de sortie finale (image_size) pour TOUS
    les samples du batch -- s'il était appliqué avec p<1.0, les samples non
    transformés garderaient leur taille d'origine, provoquant un crash au
    collate du DataLoader ("stack expects each tensor to be equal size").
    La variation d'augmentation vient de `scale`/`ratio` (aléatoires à
    chaque appel), pas de la probabilité d'application du resize lui-même.
    """
    try:
        return A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=scale, ratio=ratio, p=1.0)
    except (TypeError, ValueError):
        return A.RandomResizedCrop(size=image_size, scale=scale, ratio=ratio, p=1.0)


def get_train_transforms(
    image_size: tuple[int, int] = (518, 518),
    rotation: bool = True,
    flip: bool = True,
    scale_jitter: bool = True,
    brightness_contrast: bool = True,
    blur: bool = True,
    shadow_simulation: bool = True,
    cutout: bool = True,
) -> A.Compose:
    """
    Transforms d'entraînement. Applique une géométrie cohérente à
    image + depth + valid_mask (via additional_targets), et une
    photométrie appliquée uniquement à l'image.
    """
    transforms = []

    # ---- Géométrie (affecte image + depth + valid_mask) ----
    if scale_jitter:
        transforms.append(_random_resized_crop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)))
    else:
        transforms.append(A.Resize(image_size[0], image_size[1]))

    if rotation:
        transforms += [
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.4),
        ]

    if flip:
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]

    # ---- Photométrie (image RGB uniquement, jamais depth/mask) ----
    if brightness_contrast:
        transforms += [
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.2),
        ]

    if blur:
        transforms.append(
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=5),
                A.MotionBlur(blur_limit=5),
            ], p=0.3)
        )

    if shadow_simulation:
        transforms.append(
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5, p=0.3,
            )
        )

    transforms.append(A.GaussNoise(std_range=(0.02, 0.12), mean_range=(0.0, 0.0), p=0.2))

    # ---- Cutout : n'affecte QUE l'image + valid_mask, jamais la depth ----
    # fill_mask=0 sur valid_mask -> ces pixels deviennent "invalides"
    # et seront exclus de la loss, plutôt que de forcer depth=0 (faux zéro).
    if cutout:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, 32),
                hole_width_range=(1, 32),
                fill=0, fill_mask=0, p=0.3,
            )
        )

    # Filet de sécurité : garantit la taille de sortie quoi qu'il arrive en amont
    # (protège contre une future modification qui réintroduirait un resize
    # probabiliste ou une étape qui change les dimensions par erreur).
    transforms.append(A.Resize(image_size[0], image_size[1]))

    transforms.append(A.Normalize(mean=MEAN, std=STD))

    return A.Compose(
        transforms,
        additional_targets={
            "depth": "mask",       # subit la géométrie, jamais la photométrie
            "valid_mask": "mask",  # idem
        },
    )


def get_val_transforms(image_size: tuple[int, int] = (518, 518)) -> A.Compose:
    """Val/test : uniquement resize + normalisation, aucune augmentation stochastique."""
    return A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=MEAN, std=STD),
        ],
        additional_targets={
            "depth": "mask",
            "valid_mask": "mask",
        },
    )


def apply_transform(transform: A.Compose, image: np.ndarray, depth: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    Applique le pipeline aux trois éléments simultanément.

    Usage dans le Dataset (__getitem__) :
        out = apply_transform(self.transform, image, depth, valid_mask)
        image, depth, valid_mask = out["image"], out["depth"], out["valid_mask"]
    """
    out = transform(image=image, depth=depth, valid_mask=valid_mask)
    return out


if __name__ == "__main__":
    # Petit auto-test de cohérence géométrique : vérifie que image/depth/valid_mask
    # ressortent avec les mêmes dimensions spatiales après transformation.
    dummy_image = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)
    dummy_depth = (np.random.rand(600, 800) * 30).astype(np.float32)
    dummy_mask = np.ones((600, 800), dtype=np.uint8)

    train_tf = get_train_transforms()
    out = apply_transform(train_tf, dummy_image, dummy_depth, dummy_mask)

    print("image:", out["image"].shape)
    print("depth:", out["depth"].shape)
    print("valid_mask:", out["valid_mask"].shape)
    assert out["image"].shape[:2] == out["depth"].shape[:2] == out["valid_mask"].shape[:2], \
        "Désalignement géométrique image/depth/mask !"
    print("OK : cohérence géométrique vérifiée.")