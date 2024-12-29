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

# Dice coefficient function
def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

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

    image_root = args.img_root
    anomaly_source_path = args.anamolypath

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

    dataset = TrainDataset(
        image_dir=image_root,
        anomaly_source_path=anomaly_source_path,
        resize=(256, 256),
        transforms_img=transforms_img,
        transforms_mask=transforms_mask
    )

    test_dataset = TestDataset(image_dir=args.imag_non_empty_dir, mask_dir=args.masks_path)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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

        model.eval()
        model_seg.eval()

        dice_scores = []
        avg_precision_scores = []

        with torch.no_grad():
            for test_batch in testloader:
                test_images = test_batch[0].to(device)
                true_masks = test_batch[1].to(device)

                gray_rec = model(test_images)
                joined_in = torch.cat((gray_rec, test_images), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # Predicted segmentation mask
                pred_mask = out_mask_sm[:, 1, :, :].argmax(dim=1)

                # Compute Dice score for this batch
                dice = dice_coefficient(pred_mask, true_masks)
                dice_scores.append(dice.item())

                # Compute Average Precision for this batch
                true_masks_flat = true_masks.view(-1).cpu().numpy()
                pred_mask_flat = pred_mask.view(-1).cpu().numpy()
                avg_precision = average_precision_score(true_masks_flat, pred_mask_flat)
                avg_precision_scores.append(avg_precision)

        # Calculate the average Dice score and average Precision
        avg_dice_score = sum(dice_scores) / len(dice_scores)
        avg_ap_score = sum(avg_precision_scores) / len(avg_precision_scores)

        print(f"Validation Dice Score: {avg_dice_score:.4f}")
        print(f"Validation Average Precision: {avg_ap_score:.4f}")

    print("Training complete. Final model saved.")
