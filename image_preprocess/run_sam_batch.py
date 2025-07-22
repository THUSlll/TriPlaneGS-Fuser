from utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, resize_image
import os
from PIL import Image
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Directory containing PNG images")
    parser.add_argument("--save_path", required=True, help="Directory to save processed images")
    parser.add_argument("--ckpt_path", default="./checkpoints/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    args = parser.parse_args()

    # Load SAM checkpoint
    sam_predictor = sam_init(args.ckpt_path, 0)
    print("Loaded SAM checkpoint.")

    # Ensure save_path directory exists
    os.makedirs(args.save_path, exist_ok=True)

    # Process all PNG images in the image_path directory with tqdm progress bar
    png_files = [f for f in os.listdir(args.image_path) if f.endswith(".png")]
    for filename in tqdm(png_files, desc="Processing images"):
        input_image_path = os.path.join(args.image_path, filename)
        save_image_path = os.path.join(args.save_path, filename)

        if os.path.exists(save_image_path):
            print(f"Skipping {save_image_path} as it already exists.")
            continue
        input_raw = Image.open(input_image_path)
        input_raw = resize_image(input_raw, 512)
        image_sam = sam_out_nosave(
            sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
        )

        image_preprocess(image_sam, save_image_path, lower_contrast=False, rescale=True)
        print(f"Processed and saved: {save_image_path}")