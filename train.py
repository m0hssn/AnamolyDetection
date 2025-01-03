from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import torchvision.transforms as transforms
import torch
from torch import optim
from loss import SSIM, FocalLoss
from train_dataset import TrainDataset
from test_dataset import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import json
import glob
import os
import random
from validation_set import ValidationSet

def split_list(input_list):
    split_point = int(len(input_list) * 0.8)
    random.shuffle(input_list)    
    part1 = input_list[:split_point]
    part2 = input_list[split_point:]
    
    return part1, part2

def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def true_positive_rate(pred, target):
    tp = ((pred == 1) & (target == 1)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return tp / (tp + fn + 1e-6)

def f1_score(pred, target):
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return 2 * (precision * recall) / (precision + recall + 1e-6)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).to(device)
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2).to(device)

    model.apply(weights_init)
    model_seg.apply(weights_init)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()
    
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr},
                                  {"params": model_seg.parameters(), "lr": args.lr}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epochs * 0.8), int(args.epochs * 0.9)], gamma=0.2)

    transforms_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transforms_mask = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_image_paths = sorted(glob.glob(f"{args.img_root}/*.jpg"))
    train_image_paths, val_part1 = split_list(train_image_paths)

    test_image_paths = [os.path.join(args.imag_non_empty_dir, filename) for filename in os.listdir(args.imag_non_empty_dir) if filename.endswith('.png') or filename.endswith('.jpg')]
    test_image_paths, val_part2 = split_list(test_image_paths)

    val_paths = val_part1 + val_part2

    dataset = TrainDataset(
        image_paths=train_image_paths,
        anomaly_source_path=args.anamolypath,
        resize=(256, 256),
        transforms_img=transforms_img,
        transforms_mask=transforms_mask
    )

    test_dataset = TestDataset(
        image_paths=test_image_paths,
        mask_dir=args.masks_path,
        transforms_img=transforms_img,
        transforms_mask=transforms_mask
    )

    validation_dataset = ValidationSet(
        image_paths=val_paths,
        mask_dir=args.masks_path,
        transforms_img=transforms_img,
        transforms_mask=transforms_mask
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    validationloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    metrics = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch") as tepoch:
            for batch in tepoch:
                gray_batch = batch[0].to(device)
                anomaly_mask = batch[1].to(device)
                aug_gray_batch = batch[2].to(device)

                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = loss_l2(gray_rec, gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)

                loss = l2_loss + ssim_loss + segment_loss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        torch.save(model.state_dict(), f"model_reconstruct_epoch_{epoch + 1}.pth")
        torch.save(model_seg.state_dict(), f"model_seg_epoch_{epoch + 1}.pth")
        print(f"Epoch {epoch + 1} complete. Loss: {avg_epoch_loss:.4f}")

        # Validation metrics
        model.eval()
        model_seg.eval()

        dice_scores = []
        avg_precision_scores = []
        tpr_scores = []
        f1_scores = []

        with torch.no_grad():
            for test_batch in testloader:
                test_images = test_batch[0].to(device)
                true_masks = test_batch[1].to(device)

                gray_rec = model(test_images)
                joined_in = torch.cat((gray_rec, test_images), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                pred_mask = (out_mask_sm[:, 1, :, :] > 0.5).int()

                dice = dice_coefficient(pred_mask, true_masks)
                dice_scores.append(dice.item())

                true_masks_flat = true_masks.view(-1).cpu().numpy()
                pred_mask_flat = pred_mask.view(-1).cpu().numpy()

                avg_precision = average_precision_score(true_masks_flat, pred_mask_flat)
                avg_precision_scores.append(avg_precision)

                tpr = true_positive_rate(pred_mask, true_masks)
                tpr_scores.append(tpr)

                f1 = f1_score(pred_mask, true_masks)
                f1_scores.append(f1)

        avg_dice_score = sum(dice_scores) / len(dice_scores)
        avg_ap_score = sum(avg_precision_scores) / len(avg_precision_scores)
        avg_tpr = sum(tpr_scores) / len(tpr_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)

        print(f"Epoch {epoch + 1} Validation Dice: {avg_dice_score:.4f}, Avg Precision: {avg_ap_score:.4f}, TPR: {avg_tpr:.4f}, F1: {avg_f1:.4f}")

        metrics.append({
            "epoch": epoch + 1,
            "dice": avg_dice_score,
            "average_precision": avg_ap_score,
            "tpr": avg_tpr,
            "f1": avg_f1
        })

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        if (epoch + 1) % 10 == 0:
            output_dir = f"epoch_{epoch + 1}_masks"
            os.makedirs(output_dir, exist_ok=True)
            
            with torch.no_grad():
                for i, test_batch in enumerate(testloader):
                    test_images = test_batch[0].to(device)
                    gray_rec = model(test_images)
                    joined_in = torch.cat((gray_rec, test_images), dim=1)
                    out_mask = model_seg(joined_in)
                    out_mask_sm = torch.softmax(out_mask, dim=1)

                    pred_mask = out_mask_sm[:, 1, :, :]

                    for j in range(pred_mask.size(0)):
                        output_path = os.path.join(output_dir, f"mask_{i * args.batch_size + j + 1}.png")
                        output_mask = pred_mask[j].cpu().numpy().squeeze()
                        transforms.ToPILImage()(output_mask).save(output_path)

    print("Training complete. Final model saved.")
