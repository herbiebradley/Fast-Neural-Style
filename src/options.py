import argparse

class Options(object):

    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
        parser.add_argument("--base_filters", type=int, default=32, help="Number of filters in the last conv layer of the generator net, default is 32")
        parser.add_argument("--residual_blocks", type=int, default=5, help="Number of residual blocks in generator net, default is 5")
        parser.add_argument("--kernel_size", type=int, default=3, help="Size of kernel in conv layers, default is 3")
        self.parser = parser

    def parse(self, training):
        # get training/testing options
        if training:
             self.parser = self._get_train_options(self.parser)
        else:
             self.parser = self._get_test_options(self.parser)

        opt = self.parser.parse_args()
        self.print_options(opt)
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for option, value in sorted(vars(opt).items()):
            if option != "training":
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
        parser.add_argument("--batch_size", type=int, default=4, help="batch size for training, default is 4")
        parser.add_argument("--data_dir", type=str, required=True, help="train set path - should be a folder containing another folder with the training images")
        parser.add_argument("--style_image", type=str, required=True, help="path to style-image")
        parser.add_argument("--save_model_dir", type=str, required=True, help="path to folder where trained model will be saved.")
        parser.add_argument("--image_size", type=int, default=256, help="size to load training images, default is 256 X 256")
        parser.add_argument("--content_weight", type=float, default=1e5, help="weight for content-loss, default is 1e5")
        parser.add_argument("--style_weight", type=float, default=1e10, help="weight for style-loss, default is 1e10")
        parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate, default is 1e-3")
        parser.add_argument("--log_interval", type=int, default=500, help="number of batches after which the training loss is logged, default is 500")
        return parser

    def _get_test_options(self, parser):
        # test specific options
        parser.add_argument('--training', action='store_true', help='boolean for training/testing')
        parser.add_argument("--content_image", type=str, required=True, help="path to content image you want to stylize")
        parser.add_argument("--output_image", type=str, required=True, help="path for saving the output image")
        parser.add_argument("--model", type=str, required=True, help="Path to saved model to be used for stylizing the image.")
        return parser
