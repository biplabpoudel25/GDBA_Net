import argparse
import os
import csv

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import metrics
import models
import utils


def save_metrics_to_file(
    all_metrics,
    average_metrics,
    filename='all_metrics.txt'
):
    """
    Save per-image metrics, average metrics, and metric value lists.
    """

    fieldnames = [
        'filename',
        'dice',
        'iou',
        'precision',
        'recall',
        'specificity'
    ]

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Save individual image metrics
        for image_metrics in all_metrics:
            writer.writerow(image_metrics)

        # Add a separator
        writer.writerow({
            key: '-' * 10 for key in fieldnames
        })

        # Add average metrics
        average_row = {'filename': 'AVERAGE'}

        for key, value in average_metrics.items():
            average_row[key] = f"{value:.4f}"

        writer.writerow(average_row)

        # Save lists of all metric values
        metric_names = [
            'dice',
            'iou',
            'precision',
            'recall',
            'specificity'
        ]

        for metric_name in metric_names:
            values = [
                image_metrics[metric_name]
                for image_metrics in all_metrics
            ]

            f.write(f"\n{metric_name.upper()} List:\n")
            f.write(
                ", ".join(
                    f"{value:.4f}" for value in values
                )
                + "\n"
            )


def save_binary_mask(pred_mask, filename, save_dir):
    """
    Save binary mask as PNG image
    """
    # Convert to uint8 and scale to 0-255 for proper visualization
    pred_mask_vis = (pred_mask * 255).astype(np.uint8)
    
    # Remove channel dimension if present
    if len(pred_mask_vis.shape) == 3 and pred_mask_vis.shape[0] == 1:
        pred_mask_vis = pred_mask_vis.squeeze(0)
    elif len(pred_mask_vis.shape) == 3 and pred_mask_vis.shape[2] == 1:
        pred_mask_vis = pred_mask_vis.squeeze(2)
    
    save_path = os.path.join(save_dir, f"{filename}_mask.png")
    
    pil_image = Image.fromarray(pred_mask_vis)
    pil_image.save(save_path)
    

def evaluate(
    loader,
    model,
    device,
    save_dir,
    save_masks=False,
    verbose=False
):
    
    model.to(device)
    model.eval()

    mask_dir = None
    if save_masks:
        mask_dir = os.path.join(save_dir, 'Predicted_masks')
        os.makedirs(mask_dir, exist_ok=True)

    metric_averagers = {
        'dice': utils.Averager(),
        'iou': utils.Averager(),
        'precision': utils.Averager(),
        'recall': utils.Averager(),
        'specificity': utils.Averager()
    }

    pbar = tqdm(loader, leave=False, desc='Testing')

    all_metrics = []

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            for k, v in batch.items():
                if k != 'filename':
                    batch[k] = v.to(device)

            inp = batch['inp']
            gt = batch['gt']
            filenames = batch['filename']

            # Model prediction
            pred = torch.sigmoid(model.infer(inp))
            pred = (pred > 0.5).float()

            for i in range(inp.shape[0]):
                # Convert prediction and ground truth to NumPy
                pred_np = pred[i].cpu().numpy()
                gt_np = gt[i].cpu().numpy()

                # Save predicted binary mask if requested
                if save_masks:
                    save_binary_mask(
                        pred_np,
                        filenames[i],
                        mask_dir
                    )

                # Calculate evaluation metrics
                image_metrics = metrics.calculate_metrics(
                    pred_np,
                    gt_np
                )

                image_metrics['filename'] = filenames[i]
                all_metrics.append(image_metrics)

                # Update metric averages
                for key, value in image_metrics.items():
                    if key != 'filename':
                        metric_averagers[key].add(value, 1)

            if verbose:
                pbar.set_description(
                    'Test Dice: {:.4f}, IoU: {:.4f}'.format(
                        metric_averagers['dice'].item(),
                        metric_averagers['iou'].item()
                    )
                )

    average_metrics = {
        key: averager.item()
        for key, averager in metric_averagers.items()
    }

    return all_metrics, average_metrics
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate GDBA-Net model."
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
        '--test_images',
        type=str,
        default=None,
        help='Path to test images.'
    )

    parser.add_argument(
        '--test_labels',
        type=str,
        default=None,
        help='Path to test masks.'
    )

    parser.add_argument(
        '--save_masks',
        action='store_true',
        help='Save predicted binary masks.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory for saving test results.'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Override test dataset paths if provided
    if args.test_images is not None:
        config['test_dataset']['dataset']['args']['root_path_1'] = args.test_images

    if args.test_labels is not None:
        config['test_dataset']['dataset']['args']['root_path_2'] = args.test_labels

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare test dataset
    spec = config['test_dataset']

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(
        spec['wrapper'],
        args={'dataset': dataset}
    )

    num_workers = spec.get('num_workers', 4)

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    print(f"Test dataset: {len(dataset)} samples")

    # Build model
    model = models.make(config['model'])

    # Load trained GDBA-Net checkpoint
    checkpoint = torch.load(
        args.model,
        map_location='cpu'
    )

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        model_state_dict = checkpoint

    model.load_state_dict(
        model_state_dict,
        strict=True
    )

    # Create output directory
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Evaluate
    all_metrics, average_metrics = evaluate(
        loader=loader,
        model=model,
        device=device,
        save_dir=save_dir,
        save_masks=args.save_masks,
        verbose=True
    )

    # Save metrics
    save_metrics_to_file(
        all_metrics,
        average_metrics,
        filename=os.path.join(save_dir, 'all_metrics.txt')
    )

    print("\nAverage Test Metrics:")
    for k, v in average_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    print(f"\nResults saved to: {save_dir}")