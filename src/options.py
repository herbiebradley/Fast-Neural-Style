import argparse
import multiprocessing
import torch

class Options(object):

    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
        self.parser = parser

    def parse(self, training):
        # get training/testing options
        if training:
             self.parser = self._get_train_options(self.parser)
        else:
             self.parser = self._get_test_options(self.parser)

        opt = self.parser.parse_args()
        if opt.cuda and not torch.cuda.is_available():
            print("ERROR: cuda is not available, try running on CPU")
            sys.exit(1)
        self.print_options(opt)
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for option, value in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(option)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(option), str(value), comment)
        message += '----------------- End -------------------'
        print(message)

    def _get_train_options(self, parser):
        # training specific options
        parser.add_argument('--training', action='store_false', help='boolean for training/testing')
        parser.add_argument("--epochs", type=int, default=2, help="number of training epochs, default is 2")
        parser.add_argument("--batch-size", type=int, default=4, help="batch size for training, default is 4")
        parser.add_argument("--dataset", type=str, required=True, help="path to training dataset, the path should point to a folder containing another folder with all the training images")
        parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg", help="path to style-image")
        parser.add_argument("--save-model-dir", type=str, required=True, help="path to folder where trained model will be saved.")
        parser.add_argument("--checkpoint-model-dir", type=str, default=None, help="path to folder where checkpoints of trained models will be saved")
        parser.add_argument("--image-size", type=int, default=256, help="size of training images, default is 256 X 256")
        parser.add_argument("--style-size", type=int, default=None, help="size of style-image, default is the original size of style image")
        parser.add_argument("--content-weight", type=float, default=1e5, help="weight for content-loss, default is 1e5")
        parser.add_argument("--style-weight", type=float, default=1e10, help="weight for style-loss, default is 1e10")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate, default is 1e-3")
        parser.add_argument("--log-interval", type=int, default=500, help="number of images after which the training loss is logged, default is 500")
        parser.add_argument("--checkpoint-interval", type=int, default=2000, help="number of batches after which a checkpoint of the trained model will be created")
        return parser

    def _get_test_options(self, parser):
        # test specific options
        parser.add_argument('--training', action='store_true', help='boolean for training/testing')
        parser.add_argument("--content-image", type=str, required=True, help="path to content image you want to stylize")
        parser.add_argument("--content-scale", type=float, default=None, help="factor for scaling down the content image")
        parser.add_argument("--output-image", type=str, required=True, help="path for saving the output image")
        parser.add_argument("--model", type=str, required=True, help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used")
        return parser
