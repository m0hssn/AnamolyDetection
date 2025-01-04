from train import train
from load_anamoly import download_and_extract_dataset
from argparse import Namespace
from generate_masks_sam import generate
from load_sam import download_file
import warnings

if __name__ == '__main':
    warnings.filterwarnings("ignore")
    args = Namespace(
                        lr=3e-4,  # Learning rate
                        epochs=500,  # Number of epochs (the model was trained for 700 eochs in the original paper)
                        batch_size=8,  # Batch size for training
                        num_workers=4,  # Number of workers for the DataLoader
                        
                        img_root="/Empty_1",  # Path to the root directory of images
                        
                        anamolypath="anamolyset/dtd/images",  # Path to the anomaly source

                        imag_non_empty_dir='/Non-empty_1',
                        csv_file_dir='/Non-empty_1/gt.csv',
                        
                        masks_path ='/sam-masks-gen',

                        url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz',
                        dl_dir='anamolyset',

                        url_sam='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                        sam_path='sam_vit_h_4b8939.pth',
                    )

    download_file(args.url_sam, args.sam_path)    
    download_and_extract_dataset(args.url, args.dl_dir)
    generate(args)
    train(args)
