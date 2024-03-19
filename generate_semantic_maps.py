import glob
import os
from tqdm import tqdm

import SimpleITK as itk
import cv2
import numpy as np
import copy


def read_mhd_file(filename):
    """Read a mhd file and return a itk.Image object"""
    itk_img = itk.ReadImage(filename)
    img = itk.GetArrayFromImage(itk_img)
    return img


def scale_image(image):
    """Scale image to 0-255"""
    # get dimension of the image
    height, width = image.shape
    for wi in range(width):
        if image[:, wi].any():
            y = np.where(image[:, wi] > 0)[0][0]
            image[y, :, ] = 1
            break

    new_width = int(np.sqrt(height ** 2 - y ** 2) * 2)
    # resize image to new_width
    image = cv2.resize(image, (new_width, height))
    # print((y, new_width, height))
    # print(((new_width / 2) ** 2 + y ** 2) - height ** 2)
    return image


def mask_triangle(y, h, w):
    v1 = np.array([0, 0])
    v2 = np.array([0, y])
    v3 = np.array([w / 2, 0])

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    det = (v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])
    mask = det >= 0
    flip_mask = np.fliplr(mask)
    combined_mask = np.logical_and(flip_mask, mask)
    return combined_mask


def generate_fan_cone(image):
    """Generate a fan cone from a 2D image"""
    height, width = image.shape
    for wi in range(width):
        if image[:, wi].any():
            y = np.where(image[:, wi] > 0)[0][0]
            image[y, :, ] = 1
            break

    new_width = int(np.sqrt(height ** 2 - y ** 2) * 2)
    # resize image to new_width
    x_grid, y_grid = np.meshgrid(np.arange(new_width), np.arange(height))
    xy_grid = np.stack([x_grid, y_grid], axis=-1)
    center = np.array([new_width / 2, 0])
    radius = np.sqrt(np.sum((xy_grid - center) ** 2, axis=-1))
    cone = (radius <= height)
    triangle = mask_triangle(y, height, width)
    cone = np.logical_and(cone, triangle).astype(np.float32)
    cone = cv2.resize(cone, (width, height), interpolation=cv2.INTER_NEAREST)
    return cone


if __name__ == "__main__":
    data_folder = "../../"
    images = []
    for folder in tqdm(glob.glob(os.path.join(data_folder, "images/*"))):
        cone_mask = None
        background_cone_mask = None
        folder_name = os.path.basename(folder)
        for image_path in glob.glob(os.path.join(data_folder, f"images/{folder_name}/*.jpg"), recursive=True):
            mask_path = image_path.replace("images", "seg_maps").replace(".jpg", ".png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if cone_mask is None:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = image / 255.0
                image = scale_image(image)
                cone_mask = generate_fan_cone(image)
                cone_mask = cv2.resize(cone_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                background_cone_mask = (cone_mask == 0)

            new_mask = copy.deepcopy(cone_mask)
            new_mask[mask > 0] = mask[mask > 0] + 1
            new_mask[background_cone_mask] = 0
            new_mask = new_mask.astype(np.uint8)
            target_path = mask_path.replace("seg_maps", "seg_maps_cone")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            cv2.imwrite(mask_path.replace("seg_maps", "seg_maps_cone"), new_mask)
            # print(np.unique(new_mask))
            # print(new_mask.dtype)
    #         import matplotlib
    #         colored_mask = matplotlib.cm.get_cmap('viridis')(new_mask/np.max(new_mask))
    #         colored_mask = colored_mask[:, :, :3]
    #         colored_mask = colored_mask * 255
    #         colored_mask = colored_mask.astype(np.uint8)
    #         images.append(colored_mask)
        # break

    # # save to gif
    # import imageio
    # imageio.mimsave('fan_cone.gif', images, duration=0.5)
