import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import models


class InferenceDataset(Dataset):
    def __init__(self, image_dir, inp_size):
        self.image_dir = image_dir

        valid_extensions = (
            '.png',
            '.jpg',
            '.jpeg',
            '.tif',
            '.tiff',
            '.bmp'
        )

        self.filenames = sorted([
            filename
            for filename in os.listdir(image_dir)
            if filename.lower().endswith(valid_extensions)
        ])

        if len(self.filenames) == 0:
            raise ValueError(
                f'No supported image files found in: {image_dir}'
            )

        self.img_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(image_path).convert('RGB')
        image = self.img_transform(image)

        filename_without_extension = os.path.splitext(
            filename
        )[0]

        return {
            'inp': image,
            'filename': filename_without_extension
        }


def save_binary_mask(pred_mask, filename, save_dir):
    """
    Save a predicted binary mask as a PNG image.
    """
    pred_mask_vis = (pred_mask * 255).astype(np.uint8)

    if len(pred_mask_vis.shape) == 3 and pred_mask_vis.shape[0] == 1:
        pred_mask_vis = pred_mask_vis.squeeze(0)
    elif len(pred_mask_vis.shape) == 3 and pred_mask_vis.shape[2] == 1:
        pred_mask_vis = pred_mask_vis.squeeze(2)

    save_path = os.path.join(
        save_dir,
        f'{filename}_mask.png'
    )

    pil_image = Image.fromarray(pred_mask_vis)
    pil_image.save(save_path)


def run_inference(loader, model, device, save_dir):
    """
    Run GDBA-Net inference and save predicted binary masks.
    """
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    pbar = tqdm(
        loader,
        leave=False,
        desc='Inference'
    )

    with torch.no_grad():
        for batch in pbar:
            inp = batch['inp'].to(device)
            filenames = batch['filename']

            pred = torch.sigmoid(
                model.infer(inp)
            )
            pred = (pred > 0.5).float()

            for i in range(inp.shape[0]):
                pred_np = pred[i].cpu().numpy()

                save_binary_mask(
                    pred_np,
                    filenames[i],
                    save_dir
                )


def main():
    parser = argparse.ArgumentParser(
        description='Run inference using a trained GDBA-Net model.'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/gdba_net.yaml',
        help='Path to the YAML configuration file.'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained GDBA-Net checkpoint.'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the directory containing input images.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='predictions',
        help='Directory where predicted masks will be saved.'
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(
            f,
            Loader=yaml.FullLoader
        )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f'Device: {device}')

    test_spec = config['test_dataset']

    inp_size = test_spec['wrapper']['args']['inp_size']
    batch_size = test_spec['batch_size']
    num_workers = test_spec.get('num_workers', 4)

    dataset = InferenceDataset(
        image_dir=args.input,
        inp_size=inp_size
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    print(f'Inference dataset: {len(dataset)} images')

    model = models.make(
        config['model']
    )

    checkpoint = torch.load(
        args.model,
        map_location='cpu'
    )

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(f'Model loaded from: {args.model}')

    run_inference(
        loader,
        model,
        device,
        args.output_dir
    )

    print(f'Predicted masks saved to: {args.output_dir}')


if __name__ == '__main__':
    main()