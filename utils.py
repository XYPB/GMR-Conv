import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt


def one_hot(x, num_classes):
    x = x.squeeze()
    out = np.zeros([x.shape[0], num_classes]).astype(int)
    out[np.arange(x.shape[0]), x] = 1
    return out


def manual_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


# @profile
def log_train_val(
    train_loss,
    test_loss=None,
    train_acc=None,
    test_acc=None,
    grad_norm=None,
    log_dir="./log",
):
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(train_loss)), train_loss, label="train_loss")
    if test_loss is not None:
        plt.plot(np.arange(len(test_loss)), test_loss, label="test_loss")
    dest = os.path.join(log_dir, "loss.png")
    plt.legend()
    plt.savefig(dest, dpi=200)
    plt.close("all")

    if train_acc is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(train_acc)), train_acc, label="train_acc")
        if test_acc is not None:
            plt.plot(np.arange(len(test_acc)), test_acc, label="test_acc")
        dest = os.path.join(log_dir, "acc.png")
        plt.legend()
        plt.savefig(dest, dpi=200)
        plt.close("all")

    if grad_norm is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(grad_norm)), grad_norm, label="grad_norm")
        dest = os.path.join(log_dir, "grad_norm.png")
        plt.legend()
        plt.savefig(dest, dpi=200)
        plt.close("all")

        plt.figure()
        plt.hist(grad_norm, 10, label="grad_norm_hist")
        percentile_90 = np.percentile(grad_norm, 90)
        dest = os.path.join(log_dir, f"grad_norm_hist_{percentile_90:.4f}.png")
        plt.legend()
        plt.savefig(dest, dpi=200)
        plt.close("all")


def log_img(ep, batch_idx, img, label=None, log_dir="./logs"):
    dest = os.path.join(log_dir, f"ep_{ep}_b_{batch_idx}.png")
    if label is not None:
        dest = dest.replace(".png", f"_label_{label}.png")
    img = (img - img.min()) / (img.max() - img.min())
    if len(img.shape) == 3:
        img = img.transpose((1, 2, 0))
    plt.imsave(dest, img)



def create_circle_mask(image_shape, radius):
    # Create grid of coordinates
    x, y = np.ogrid[: image_shape[0], : image_shape[1]]

    center = (image_shape[0] // 2, image_shape[1] // 2)
    # Calculate squared distance from each point to the center
    distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # Create mask where True denotes points inside the circle
    mask = distance_squared <= radius**2

    return mask.astype(np.uint8)



def acc_at_topk(targets, prob, k=5):
    assert len(targets) == len(prob)
    topk = np.argsort(prob, axis=1)[:, -k:]
    correct = 0
    for i in range(len(targets)):
        if targets[i] in topk[i]:
            correct += 1
    return correct / len(targets)


def force_cudnn_initialization():
    s = 32
    dev = torch.device("cuda")
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
    )
