import argparse

parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
# Training settings
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=5,
    metavar="N",
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
)
parser.add_argument(
    "--min-lr", type=float, default=1e-6, help="minimal learning rate (default: 1.0)"
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-3,
    metavar="WD",
    help="weight decay (default: 1e-3)",
)
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="MT", help="momentum (default: 0.9)"
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    metavar="NW",
    help="number of workers (default: 8)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.7,
    metavar="M",
    help="Learning rate step gamma (default: 0.7)",
)
parser.add_argument(
    "--step-size", type=float, default=10, help="Step size to update learning rate"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)
parser.add_argument(
    "--eval-only", action="store_true", default=False, help="quickly evaluate the model"
)
parser.add_argument(
    "--adam", action="store_true", default=False, help="using adam optim"
)
parser.add_argument(
    "--adamW", action="store_true", default=False, help="using adamW optim"
)
parser.add_argument("--sgd", action="store_true", default=False, help="using SGD optim")
parser.add_argument(
    "--cos", action="store_true", default=False, help="using cosine scheduler"
)
parser.add_argument(
    "--step", action="store_true", default=False, help="using step scheduler"
)
parser.add_argument(
    "--step_scheduler-size",
    type=int,
    default=1,
    help="step size for step scheduler (default: 1)",
)
parser.add_argument(
    "--multi-step",
    action="store_true",
    default=False,
    help="using multi-step scheduler",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--model-type", type=str, default="conv", help="which kind of model to use"
)
parser.add_argument(
    "--cudnn",
    action="store_true",
    default=False,
    help="use cudnn benchmark to speed up the training",
)
parser.add_argument("--dev", action="store_true", default=False, help="debug mode")
parser.add_argument("--log-img", action="store_true", default=False, help="log images")
parser.add_argument(
    "--amp", action="store_true", default=False, help="use AMP training"
)
parser.add_argument(
    "--scaler", action="store_true", default=False, help="use gradient scaler training"
)
parser.add_argument(
    "--bf16", action="store_true", default=False, help="use bfloat16 for training"
)
parser.add_argument(
    '--ddp', action='store_true', default=False, help='use DDP training'
)
parser.add_argument(
    '--world-size', type=int, default=1, help='world size for ddp training'
)
parser.add_argument(
    '--dist-url', type=str, default='tcp://localhost:10001', help='communication url for ddp training'
)

# Log settings
parser.add_argument(
    "--base-log-dir", type=str, default="./logs", help="log output directory"
)
parser.add_argument(
    "--log", action="store_true", default=False, help="log output and train images"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=200,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--save-model",
    action="store_true",
    default=False,
    help="For saving the current Model",
)
parser.add_argument(
    "--log-grad-norm",
    action="store_true",
    default=False,
    help="log model gradient norm",
)
parser.add_argument("--exp", type=str, default="task", help="name of the experiments")
parser.add_argument(
    "--save-rec",
    action="store_true",
    default=False,
    help="For saving the current Model recurrently",
)
parser.add_argument(
    "--save-best",
    action="store_true",
    default=False,
    help="For saving the current best Model",
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=5,
    help="how many epochs to wait before save the model",
)
parser.add_argument("--load-model", type=str, default="", help="load pre-trained model")
parser.add_argument(
    "--resume", type=str, default="", help="resume from previous training"
)
parser.add_argument(
    "--cur-ep", type=int, default=None, help="epoch of previous training"
)
parser.add_argument(
    "--wandb", action="store_true", default=False, help="use wandb for logging"
)


# Augmentation
parser.add_argument(
    "--rotate", action="store_true", default=False, help="rotate the input"
)
parser.add_argument(
    "--train-rotate",
    action="store_true",
    default=False,
    help="rotate the input during training",
)
parser.add_argument(
    "--img-size", type=int, default=128, help="target image size of the input"
)
parser.add_argument(
    "--degree", type=int, default=180, help="degree to rotate during test time"
)
parser.add_argument(
    "--train-degree", type=int, default=30, help="degree to rotate during train time"
)
parser.add_argument(
    "--fix-rotate",
    action="store_true",
    default=False,
    help="rotate the input with fixed angle",
)
parser.add_argument(
    "--moco-aug", action="store_true", default=False, help="using MoCo style transforms"
)
parser.add_argument(
    "--translation", action="store_true", default=False, help="add random translation"
)
parser.add_argument(
    "--test-translation",
    action="store_true",
    default=False,
    help="add random translation",
)
parser.add_argument(
    "--translate-ratio", type=float, default=0.2, help="ratio of random translation"
)
parser.add_argument(
    "--test-translate-ratio",
    type=float,
    default=None,
    help="ratio of random translation at test time",
)
parser.add_argument(
    "--vflip", action="store_true", default=False, help="add vertical flipping"
)
parser.add_argument(
    "--hflip", action="store_true", default=False, help="add horizontal flipping"
)
parser.add_argument(
    "--dflip", action="store_true", default=False, help="add 3d flipping"
)
parser.add_argument(
    "--eval-flip",
    action="store_true",
    default=False,
    help="evaluate the model with flipping",
)
parser.add_argument(
    "--eval-interval", type=int, default=1000, help="interval of specific evaluation"
)
parser.add_argument(
    "--pad-size-h", type=int, default=None, help="height to pad the image"
)
parser.add_argument(
    "--pad-size-w", type=int, default=None, help="weight to pad the image"
)
parser.add_argument(
    "--train-flip-p",
    type=float,
    default=0.5,
    help="probability of apply flipping during training",
)
parser.add_argument(
    "--affine-prob",
    type=float,
    default=0.2,
    help="probability of apply affine transform during training",
)
parser.add_argument(
    "--scale-ratio", type=float, default=0.1, help="ratio to scale the image"
)
parser.add_argument(
    "--shear-ratio", type=float, default=0.1, help="ratio to shear the image"
)
parser.add_argument(
    "--expand",
    action="store_true",
    default=False,
    help="expand the image during rotation",
)
parser.add_argument(
    "--padding",
    type=str,
    default="zeros",
    help="padding method for rotation: zeros | reflect",
)
parser.add_argument(
    "--aug-axis", type=str, default="x", help="axis to apply augmentation: x | y | z"
)
parser.add_argument(
    "--interpolation",
    type=str,
    default="nearest",
    help="interpolation method for rotation: nearest | bilinear | bicubic",
)


# GMR conv settings
parser.add_argument(
    "--gmr-shape", type=str, default="o", help="shape of the RI conv kernel"
)
parser.add_argument(
    "--gmr-conv-size", type=int, default=3, help="size of the RI convolution kernel"
)
parser.add_argument(
    "--gmr-conv-k",
    nargs="+",
    default=None,
    help="# of layers of the RI convolution kernel",
)
parser.add_argument(
    "--res-inplanes",
    type=int,
    default=64,
    help="use different convolution channel size",
)
parser.add_argument(
    "--res-keep-conv1",
    action="store_true",
    default=False,
    help="Not to down sample for resnet Conv1",
)

# Model settings
parser.add_argument(
    "--dropout-p", type=float, default=-1, help="probability of dropout layers"
)
parser.add_argument(
    "--large-conv",
    action="store_true",
    default=False,
    help="Use 7x7 convolution layers",
)
parser.add_argument(
    "--maxpool", action="store_true", default=False, help="use max pooling"
)


# MedMNIST args
parser.add_argument(
    "--gmr-conv-size-list",
    nargs="+",
    default=None,
    help="Size of each MNIST ri conv kernel",
)
parser.add_argument(
    "--eval-rot",
    action="store_true",
    default=False,
    help="Evaluate trained model on different rotation degrees",
)
parser.add_argument(
    "--med-mnist", type=str, default="", help="indicate the medMNIST dataset to be used"
)
parser.add_argument(
    "--no-aug", action="store_true", default=False, help="Train w/o augmentation"
)
parser.add_argument(
    "--multi-label",
    action="store_true",
    default=False,
    help="Train with multi-label loss",
)
parser.add_argument(
    "--medmnist-size", type=int, default=28, help="Size of MedMNIST image"
)


def get_opt():
    args = parser.parse_args()
    if args.gmr_conv_size_list != None:
        args.gmr_conv_size_list = [int(x) for x in args.gmr_conv_size_list]
    if isinstance(args.gmr_conv_k, list):
        args.gmr_conv_k = [int(x) for x in args.gmr_conv_k]
    return args
