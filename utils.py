import os
from PIL import Image
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import rotate
import imageio
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d


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


def vis_num_ringsernel(
    gmr_conv, vis_step=1, mode="2D", vmin=None, vmax=None, in_start=0, out_start=0
):
    weight = gmr_conv.weight.detach().cpu().numpy()
    kernel = gmr_conv._make_weight_matrix(gmr_conv.weight).detach().cpu().numpy()
    Cout, Cin, k = weight.shape
    kernel_size = kernel.shape[-1]
    if isinstance(vis_step, int):
        in_vis_step = vis_step
        out_vis_step = vis_step
    else:
        in_vis_step, out_vis_step = vis_step

    in_channels = np.arange(in_start, Cin, in_vis_step)
    out_channels = np.arange(out_start, Cout, out_vis_step)

    if mode == "2D":
        plt.figure(figsize=(2 * len(out_channels), 2 * len(in_channels)))
        idx = 1
        for cin in in_channels:
            for cout in out_channels:
                ax = plt.subplot(len(in_channels), len(out_channels), idx)
                # plt.title(f'K_{cin}_{cout}')
                plt.imshow(kernel[cout, cin, :, :], cmap="Blues", vmin=vmin, vmax=vmax)
                plt.axis("off")
                # cbar = plt.colorbar()

                # if vmin is None:
                #     vmin = kernel[cout, cin, :, :].min()
                # if vmax is None:
                #     vmax = kernel[cout, cin, :, :].max()
                # Apply scientific notation to colorbar
                # # cbar.set_ticks([vmin, (vmin + vmax)/2, vmax])
                # # cbar.set_ticklabels([vmin, (vmin + vmax)/2, vmax])  # Custom labels
                # formatter = ScalarFormatter(useMathText=True)
                # formatter.set_scientific(True)
                # # formatter.set_powerlimits((vmin, vmax))  # You can adjust these limits
                # cbar.ax.yaxis.set_major_formatter(formatter)

                idx += 1
    else:
        plt.figure(figsize=(2 * len(in_channels), 2 * len(out_channels)))
        x = np.arange(k)
        idx = 1
        for cin in in_channels:
            for cout in out_channels:
                ax = plt.subplot(len(in_channels), len(out_channels), idx)
                plt.title(f"K_{cin}_{cout}")
                plt.ylim((vmin, vmax))
                ax.plot(x, weight[cout, cin, :])
                idx += 1
    return plt.gcf()


def create_circle_mask(image_shape, radius):
    # Create grid of coordinates
    x, y = np.ogrid[: image_shape[0], : image_shape[1]]

    center = (image_shape[0] // 2, image_shape[1] // 2)
    # Calculate squared distance from each point to the center
    distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # Create mask where True denotes points inside the circle
    mask = distance_squared <= radius**2

    return mask.astype(np.uint8)


def vis_cam(
    nrow,
    ncol,
    model,
    target_layers,
    input_list,
    test_dataset=None,
    img_idx=None,
    save_path=None,
    vis_output=False,
    vis_input=False,
    angle_list=None,
    rot_back=False,
    circle_mask=False,
):
    plt.figure(figsize=(2 * ncol, 2 * nrow))
    plt.margins(x=0)
    for i in range(nrow * ncol):
        if input_list is not None:
            input, label = input_list[i]
        else:
            input, label = test_dataset.__getitem__(img_idx[i])
        if vis_input:
            plt.imsave(
                save_path.replace(".png", f"_{i}_input.png"),
                input.numpy().transpose((1, 2, 0)),
            )
        cam = GradCAM(model=model, target_layers=target_layers)
        input_tensor = input[None]
        targets = [ClassifierOutputTarget(label)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        img = input.numpy().transpose((1, 2, 0))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)

        if angle_list is not None and rot_back:
            angle = angle_list[i]
            h, w, _ = visualization.shape
            visualization = rotate(visualization, -angle, reshape=False)
        if circle_mask:
            h, w, _ = visualization.shape
            mask = create_circle_mask((h, w), 0.5 * h)
            mask = mask == 0
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            visualization = np.ma.masked_array(visualization, mask)
            visualization[mask] = 255

        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(visualization)
        plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


outputs = []


def hook(module, input, output):
    global outputs
    outputs.clear()
    outputs.append(output)


outputs = []


def hook_split(module, input, output):
    global outputs
    outputs.clear()
    final_output, temp_output = module._costume_forward_dw(input[0], return_all=True)
    outputs.append(final_output)
    outputs.append(temp_output)


def vis_feat(
    nrow,
    ncol,
    model,
    target_layer,
    input_list,
    feat_idx=1,
    test_dataset=None,
    img_idx=None,
    save_path=None,
    vis_output=False,
    vis_input=False,
    angle_list=None,
    rot_back=False,
    circle_mask=False,
    create_gif=False,
    frame_duration=0.0005,
    gif_input=False,
    rescale=True,
    split_plot=False,
):
    if create_gif:
        img_list = []
    else:
        plt.figure(figsize=(2 * ncol, 2 * nrow))
        plt.margins(x=0)
    for i in range(nrow * ncol):
        if input_list is not None:
            input, label = input_list[i]
        else:
            input, label = test_dataset.__getitem__(img_idx[i])
        if split_plot:
            handle = getattr(model, target_layer)[1].conv1.register_forward_hook(
                hook_split
            )
        elif "layer" in target_layer:
            handle = getattr(model, target_layer)[1].conv1.register_forward_hook(hook)
        else:
            handle = getattr(model, target_layer).register_forward_hook(hook)
        if vis_input:
            plt.imsave(
                save_path.replace(".png", f"_{i}_input.png"),
                input.numpy().transpose((1, 2, 0)),
            )
        output = model(input[None])
        input_img = input[None].detach().numpy().squeeze().transpose((1, 2, 0))
        if split_plot:
            num_rings = outputs[1].shape[1]
            for j in range(num_rings):
                ring_img = outputs[1][:, j].detach().numpy().squeeze()
                ring_img = ring_img.mean(0)
                ring_img = (ring_img - np.min(ring_img)) / (
                    np.max(ring_img) - np.min(ring_img)
                )
                plt.imsave(
                    save_path.replace(".png", f"_{i}_ring_{j}.png"),
                    ring_img,
                    cmap="viridis",
                )
        feat = outputs[0].detach().numpy().squeeze()
        if feat_idx == "mean":
            img = feat.mean(0)
        else:
            assert isinstance(feat_idx, int)
            img = feat[feat_idx]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        if angle_list is not None and rot_back:
            angle = angle_list[i]
            h, w = img.shape
            input_img = rotate(input_img, -angle, reshape=False)
            img = rotate(img, -angle, reshape=False)
        if circle_mask:
            h, w = img.shape
            mask = create_circle_mask(img.shape, 0.5 * h)
            mask = mask == 0
            img = np.ma.masked_array(img, mask)
            h, w = input_img.shape[:2]
            mask = create_circle_mask((h, w), 0.5 * h)
            mask = mask == 0
            mask = mask.repeat(3).reshape(h, w, 3)
            input_img = np.ma.masked_array(input_img, mask)

        if create_gif:
            print(f"Creating gif frame: {i+1}/{nrow*ncol}\r", end="")
            if gif_input:
                img_list.append(input_img)
            else:
                # print(img.shape)
                img_list.append(img)
        else:
            plt.subplot(nrow, ncol, i + 1)
            plt.imshow(img, cmap="viridis")
            plt.axis("off")
            plt.imsave(
                save_path.replace(".png", f"_{i}.png"), img, dpi=300, cmap="viridis"
            )
        # plt.imsave(save_path.replace(".png", f"_{i}.png"), img, dpi=300, cmap='viridis')
        handle.remove()
        if vis_output:
            plt.figure()
            plt.bar(
                np.arange(10), torch.softmax(output[0], 0).detach().numpy().squeeze()
            )
            plt.ylim((0, 1))
            plt.savefig(
                save_path.replace(".png", f"_{i}_output.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close("all")
            plt.cla()
    if create_gif:
        cmap = plt.cm.viridis
        output_path = save_path.replace(".png", ".gif")
        # global rescale
        max_intensity = max([img.max() for img in img_list])
        min_intensity = min([img.min() for img in img_list])
        h, w = img_list[0].shape[:2]
        mask = create_circle_mask((h, w), 0.5 * h)
        mask = mask == 0
        if not gif_input:
            mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
        with imageio.get_writer(
            output_path, mode="I", duration=frame_duration
        ) as writer:
            for img in img_list:
                if rescale:
                    img = (img - min_intensity) / (max_intensity - min_intensity)
                if gif_input:
                    image_color = img
                else:
                    image_color = cmap(img)
                image_color = (image_color * 255).astype(np.uint8)
                # fill masked region with white color
                image_color[mask] = 255
                writer.append_data(image_color)
        img_stack = np.stack(img_list, axis=0)  # N, H, W
        rotation_wise_std = np.std(img_stack, axis=0)
        # skip the masked region
        if gif_input:
            rotation_wise_std[mask] = 0
        else:
            rotation_wise_std[mask[:, :, 0]] = 0
        avg_std = np.mean(rotation_wise_std)
        print(f"Avg. rotation-wise signal STD: {avg_std:.4f}\n")
        return rotation_wise_std
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return None


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


def visualize_pca_2d(features, labels, vis=True):
    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    features = features.astype(np.float64)
    # features = (features - features.mean(axis=0)) / features.std(axis=0)
    features = features / np.linalg.norm(features, axis=1)[:, None]
    pca_result = pca.fit_transform(features)
    if not vis:
        return None, pca_result

    # Create a scatter plot of the PCA results
    plt.figure(figsize=(10, 8))
    classes = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, class_label in enumerate(classes):
        idx = labels == class_label
        plt.scatter(
            pca_result[idx, 0], pca_result[idx, 1], color=colors[i], label=class_label
        )

    plt.title("PCA Visualization of Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.grid(True)
    return plt.gcf(), pca_result


def visualize_pca_3d(features, labels, vis=True):
    # Perform PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(features)
    if not vis:
        return None, X_pca

    # Visualize in 3D space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    classes = np.unique(labels)
    colors = plt.cm.tab10(
        np.linspace(0, 1, len(classes))
    )  # Assuming there are 3 classes
    for class_label in classes:
        class_indices = np.where(labels == class_label)[0]
        ax.scatter(
            X_pca[class_indices, 0],
            X_pca[class_indices, 1],
            X_pca[class_indices, 2],
            color=colors[class_label],
            label=f"Class {class_label}",
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    plt.title("3D PCA Visualization")
    plt.show()


def visualize_tsne_2d(features, labels, vis=True):
    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        n_iter=1500,
        n_iter_without_progress=300,
        verbose=1,
        perplexity=30,
    )
    # tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto',
    #             n_iter=1500, n_iter_without_progress=300, verbose=1, perplexity=20)
    # features = features.astype(np.float64)
    # features = features / np.linalg.norm(features, axis=1)[:, None]
    tsne_result = tsne.fit_transform(features)
    if not vis:
        return None, tsne_result

    # Create a scatter plot of the t-SNE results
    plt.figure(figsize=(10, 8))
    classes = np.unique(labels)
    classes = np.sort(classes)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, class_label in enumerate(classes):
        idx = labels == class_label
        plt.scatter(
            tsne_result[idx, 0],
            tsne_result[idx, 1],
            color=colors[i],
            label=class_label,
            s=5,
        )

    plt.title("t-SNE Visualization of Data")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.xlim(-110, 110)
    plt.ylim(-110, 110)
    plt.grid(True)
    return plt.gcf(), tsne_result


def visualize_tsne_3d(features, labels, vis=True):
    # Perform t-SNE
    tsne = TSNE(
        n_components=3,
        random_state=42,
        init="pca",
        learning_rate="auto",
        n_iter=1500,
        n_iter_without_progress=300,
        verbose=1,
        perplexity=30,
    )
    # features = features.astype(np.float64)
    # features = features / np.linalg.norm(features, axis=1)[:, None]
    tsne_result = tsne.fit_transform(features)
    if not vis:
        return None, tsne_result

    # Create a scatter plot of the t-SNE results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    classes = np.unique(labels)
    classes = np.sort(classes)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, class_label in enumerate(classes):
        idx = labels == class_label
        plt.scatter(
            tsne_result[idx, 0],
            tsne_result[idx, 1],
            tsne_result[idx, 2],
            color=colors[i],
            label=class_label,
            s=5,
        )

    plt.title("3D tSNE Visualization")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_zlim(-110, 110)
    return plt.gcf(), tsne_result


def interpolate_1d_array(array, K, axis=1, interp_type="linear"):
    """
    Interpolates a 1D array of size N into K evenly spaced points.

    Parameters:
    array (ndarray): Input ND array of size N.
    K (int): Number of points to interpolate.
    axis (int): Axis along which to interpolate.

    Returns:
    ndarray: Interpolated 1D array of size K.
    """
    N = array.shape[axis]
    x = np.linspace(0, N - 1, N)  # Original indices
    xp = np.linspace(0, N - 1, K)  # New indices

    # Create interpolation function
    interpolator = interp1d(x, array, axis=axis, kind=interp_type)

    # Interpolate the array
    interpolated_array = interpolator(xp)

    return interpolated_array


def create_logits_gif(
    predictions, label, degrees, label2cap, color="tab:blue", dest="./tmp/logits.gif"
):
    n_classes = len(label2cap)
    fig_list = []
    for idx, deg in enumerate(degrees):
        cur_pred = predictions[idx]
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(n_classes), cur_pred, color=color)
        plt.xticks(np.arange(n_classes), label2cap, rotation=15)
        plt.title(f"Predictions at {deg} degrees")
        plt.ylim((0, 1))
        plt.xlim((-0.5, n_classes - 0.5))
        fig = plt.gcf()
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer._renderer)
        plt.close("all")
        fig_list.append(image_array)
    with imageio.get_writer(dest, mode="I", duration=0.002) as writer:
        for img in fig_list:
            writer.append_data(img)
    del fig_list
