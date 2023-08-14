import torch
import argparse
from pytorch_lightning import LightningModule, Trainer
from train import SIRA_Module
from datamodule import SIRA_DataModule
import time


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='unspecified',
                        type=str, help='experiment name')
    parser.add_argument('--ckpt', default='', type=str,
                        help='filepath of weights')
    parser.add_argument('--trainset', default='flickr_train_10k',
                        type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='flickr',
                        type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='/mnt/ssd/datasets/flickr-soundnet/train/',
                        type=str, help='Root directory path of data')
    parser.add_argument('--test_data_path', default='/mnt/ssd/datasets/flickr-soundnet/test_flow_v4/',
                        type=str, help='Root directory path of data')
    parser.add_argument('--image_size', default=224,
                        type=int, help='Height and width of inputs')
    parser.add_argument(
        '--gt_path', default='/mnt/ssd/datasets/flickr-soundnet/Dataset/Annotations/', type=str)
    parser.add_argument('--epsilon', default=0.65, type=float)
    parser.add_argument('--epsilon_margin', default=0.25, type=float)
    parser.add_argument('--logit_temperature', default=0.07, type=float)
    parser.add_argument('--trimap', default=1, type=int)
    parser.add_argument('--pretrain_flow', default=1, type=int)
    parser.add_argument('--pretrain_vision', default=1, type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--flow', type=int, default=1)
    parser.add_argument('--flowtype', type=str, default='cnn')
    parser.add_argument('--freeze_vision', type=int, default=1)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    trainer = Trainer(accelerator='gpu',
                      devices="auto")

    datamodule = SIRA_DataModule(args)

    if '.pth' in args.ckpt:
        module = SIRA_Module(args)

        print(module, "loaded model!")

        trainer.test(module, datamodule)
    else:
        model = SIRA_Module.load_from_checkpoint(
            checkpoint_path=args.ckpt,
            map_location=None,
        )

        model.args = args

        print(model, "loaded model!")

        trainer.test(model, datamodule, ckpt_path=args.ckpt)
