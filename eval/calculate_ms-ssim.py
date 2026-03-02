import os
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision.transforms import ToTensor
from tqdm import tqdm


def open_and_preprocess(img_path, transform=None):
    """Preprocesses an image by opening it and applying a transform"""
    img = Image.open(img_path).convert("L")
    if transform is not None:
        return transform(img)
    return img

def calc_intraprompt_diversity(
        input_folder,
        n_images=2000,
        n_repetitions=4,
        search_pattern="*_0.jpg"
    ):
    """
    Calculates the MS-SSIM pairwise between all corresponding images in a folder.
    The folder must contain n_repetitions images for each dicom_id.
    The images must be named dicom_id_0.jpg, dicom_id_1.jpg, dicom_id_2.jpg, dicom_id_3.jpg

    Args:
        input_folder (str): Path to folder containing images
        n_images (int): Number of images to process
        n_repetitions (int): Number of repetitions per image
        search_pattern (str): Search pattern for glob

    Returns:
        results (pd.DataFrame): Data frame with n_repetitions MS-SSIM values per image
    """
    assert os.path.exists(input_folder), f"Folder doesnt exist: {input_folder}"
    assert n_repetitions > 1, "n_repetitions must be > 1 for pairwise comparison"

    # Detect files to process
    flist = list(tqdm(Path(input_folder).glob(search_pattern)))
    assert len(flist) > 0, "No files found"
    if(len(flist)) < n_images:
        n_images = len(flist)
        print(f"Folder contained only {len(flist)} files, set n_images to {n_images}.")
    else:
        print(f"Processing {n_images} in folder {input_folder}...")
    
    # Create comparisons
    comparisons = list(combinations(range(n_repetitions), 2)) # [(0, 1), (0, 2)...]
    x_imgs, y_imgs = list(zip(*comparisons)) # [(0, 0, 0, 1, 1, 2), (1, 2, 3, 2, 3, 3)]

    print(f"Comparisons to run ({len(comparisons)}): {comparisons}")

    # Create transform
    TT = ToTensor() # Auto-scales to [0.0,1.0] if input is outside that range

    # Test first image
    first_img = open_and_preprocess(flist[0].as_posix(), transform=TT)
    print(f"First image stats: shape: {first_img.shape}, " \
          +f"min:{torch.min(first_img):.3f}, " \
          +f"max:{torch.max(first_img):.3f}, " \
          +f"mean:{torch.mean(first_img):.3f}")

    results = []

    # Iterate over n_images files from flist and calculate pair-wise MS-SSIM
    for img_path in tqdm(flist[:n_images], total=n_images):
        # img_list = [img_path.as_posix().replace("_0.jpg", f"_{i}.jpg") for i in range(n_repetitions)]
        img_list = sorted(Path(input_folder).glob(search_pattern))
        imgs = [open_and_preprocess(img_path_, transform=TT) for img_path_ in img_list]

        X = torch.stack([imgs[i] for i in x_imgs])
        Y = torch.stack([imgs[k] for k in y_imgs])

        # Calculate MS-SSIM and append results
        results.append([
            img_path.stem[:-2],  # dicom_id_0.jpg -> dicom_id
            *ms_ssim(X, Y, data_range=1.0, size_average=False).tolist()
        ])


    # Put results into a pandas data frame
    results = pd.DataFrame(results, columns=[
                                    "dicom_id",
                                    *[f"ms_ssim_{i}" for i in range(len(x_imgs))]
                                    ])
    return results

def calc_single_prompt_diversity(input_folder, n_images=2000, search_pattern="*.png", max_pairs=None, batch_j=64):

    from random import sample
    TT = ToTensor()
    files = sorted(list(Path(input_folder).glob(search_pattern)))[:n_images]
    assert len(files) >= 2, "Need at least 2 images"

    from itertools import combinations
    all_pairs = list(combinations(range(len(files)), 2))
    if max_pairs is not None and max_pairs < len(all_pairs):
        pair_indices = sample(all_pairs, max_pairs)
    else:
        pair_indices = all_pairs

    results = []

    for i in tqdm(range(len(files) - 1), desc="outer i"):
        img_i = TT(Image.open(files[i]).convert("L"))

        js = [j for (ii, j) in pair_indices if ii == i]

        for start in range(0, len(js), batch_j):
            chunk = js[start:start + batch_j]
            if not chunk:
                continue

            X = torch.stack([img_i] * len(chunk), dim=0)  # [B, 1, H, W]
            Y = torch.stack([TT(Image.open(files[j]).convert("L")) for j in chunk], dim=0)
            vals = ms_ssim(X, Y, data_range=1.0, size_average=False).tolist()
            for j, v in zip(chunk, vals):
                results.append((files[i].name, files[j].name, v))

    df = pd.DataFrame(results, columns=["img_i", "img_j", "ms_ssim"])
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Path to folder containing images")
    parser.add_argument("--n_images", type=int, default=2000, help="Number of images to process")
    parser.add_argument("--n_repetitions", type=int, default=4, help="Number of repetitions per image")
    parser.add_argument("--search_pattern", type=str, default="*_0.jpg", help="Search pattern for glob")
    parser.add_argument("--output_folder", type=str, default=".", help="Path to output folder")
    # parser.add_argument("--suppress_progress_bar", action="store_true", default=False)
    args = parser.parse_args()

    results = calc_intraprompt_diversity(
        input_folder=args.input_folder,
        n_images=args.n_images,
        n_repetitions=args.n_repetitions,
        search_pattern=args.search_pattern
    )

    # results = calc_single_prompt_diversity(
    #     input_folder=args.input_folder,
    #     n_images=args.n_images,
    #     search_pattern=args.search_pattern
    # )

    # Save results
    os.makedirs(args.output_folder, exist_ok=True)
    fname = f"{args.output_folder}/ms-ssim_results_{Path(args.input_folder).name}_n={len(results)}.csv"
    results.to_csv(fname, index=False)
    print(f"Saved results to {fname}")

    # Compute and print mean ± std for all ms_ssim columns 
    import numpy as np
    ms_cols = [c for c in results.columns if c.startswith("ms_ssim")]
    if len(ms_cols) > 0:
        values = results[ms_cols].to_numpy().flatten()
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"\nOverall MS-SSIM mean ± std: {mean_val:.6f} ± {std_val:.6f}")
    else:
        print("No ms_ssim columns found for mean/std computation.")

