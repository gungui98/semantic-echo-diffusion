import re
from functools import wraps

import einops
import imageio
import matplotlib
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from einops import rearrange, reduce, repeat
from torch import nn


# checking shape
# @nils-werner
# https://github.com/arogozhnikov/einops/issues/168#issuecomment-1042933838

def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


# do same einops operations on a list of tensors

def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


# do einops with unflattening of anonymously named dimensions
# (...flattened) ->  ...flattened

def _with_anon_dims(fn):
    @wraps(fn)
    def inner(tensor, pattern, **kwargs):
        regex = r'(\.\.\.[a-zA-Z]+)'
        matches = re.findall(regex, pattern)
        get_anon_dim_name = lambda t: t.lstrip('...')
        dim_prefixes = tuple(map(get_anon_dim_name, set(matches)))

        update_kwargs_dict = dict()

        for prefix in dim_prefixes:
            assert prefix in kwargs, f'dimension list "{prefix}" was not passed in'
            dim_list = kwargs[prefix]
            assert isinstance(dim_list,
                              (list, tuple)), f'dimension list "{prefix}" needs to be a tuple of list of dimensions'
            dim_names = list(map(lambda ind: f'{prefix}{ind}', range(len(dim_list))))
            update_kwargs_dict[prefix] = dict(zip(dim_names, dim_list))

        def sub_with_anonymous_dims(t):
            dim_name_prefix = get_anon_dim_name(t.groups()[0])
            return ' '.join(update_kwargs_dict[dim_name_prefix].keys())

        pattern_new = re.sub(regex, sub_with_anonymous_dims, pattern)

        for prefix, update_dict in update_kwargs_dict.items():
            del kwargs[prefix]
            kwargs.update(update_dict)

        return fn(tensor, pattern_new, **kwargs)

    return inner


# generate all helper functions

rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)

rearrange_with_anon_dims = _with_anon_dims(rearrange)
repeat_with_anon_dims = _with_anon_dims(repeat)
reduce_with_anon_dims = _with_anon_dims(reduce)

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def video_tensor_from_first_segmap(tensor, cond, path, duration=120, loop=0, optimize=True):
    tensor = einops.rearrange(tensor, 'c t h w -> t h w c')
    images = tensor.detach().cpu().numpy()
    cond = einops.repeat(cond, "h w -> t h w", t=images.shape[0])
    if exists(cond):
        cond = cond.cpu().detach().numpy()[:images.shape[0]]
        cond = cond / cond.max()
        colored_cond = matplotlib.cm.get_cmap('viridis')(cond)[..., :3]
        # where cond not 0, add weighted cond to image with opacity
        images = np.concatenate([images, colored_cond], axis=2)

    images = (images * 255).astype('uint8')
    images = list(map(Image.fromarray, images))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=loop, optimize=optimize)

def video_tensor_to_gif(tensor, cond, path, duration=120, loop=0, optimize=True, opacity=0.7):
    tensor = einops.rearrange(tensor, 'c t h w -> t h w c')
    images = tensor.detach().cpu().numpy()
    if exists(cond):
        cond = cond.cpu().detach().numpy()[:images.shape[0]]
        cond = cond / cond.max()
        colored_cond = matplotlib.cm.get_cmap('viridis')(cond)[..., :3]
        # where cond not 0, add weighted cond to image with opacity
        mask = cond > 0
        images[mask] = images[mask] * opacity + colored_cond[mask] * (1 - opacity)
    images = (images * 255).astype('uint8')
    images = list(map(Image.fromarray, images))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=loop, optimize=optimize)


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def plot_seq(volume, first_segment, save_path):
    """
    Plot the sequence of images and segmentations
    :param volume: (n, c, t, h, w)
    :param first_segment: (n, h, w)
    """
    
    vis = []
    first_image = volume[0]

    n_line = 40
    grid = np.stack(np.meshgrid(np.linspace(0, 1, n_line), np.linspace(0, 1, n_line)), axis=2)

    batch_size = volume.shape[0]
    org_size = volume.shape[-2:]
    n_image = volume.shape[2]

    imageio.mimsave(save_path, vis, loop=0)

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_seg2onehot(x, num_classes):
    x = torch.nn.functional.one_hot(x, num_classes=num_classes)
    x = rearrange(x, 'b f h w c -> b c f h w')
    x = x.float()
    return x


def convert_onehot2seg(x):
    x = rearrange(x, 'b c f h w -> b f h w c')
    x = torch.argmax(x, dim=-1)
    return x


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

def init_network_weights(m):
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)

def get_jacdet2d(displacement, grid=None, backward=False):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*2*h*w
    '''
    Dx_x = (displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1])
    Dx_y = (displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1])
    Dy_x = (displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1])
    Dy_y = (displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1])

    # Dy_x = (displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1])
    # Dy_y = (displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1])
    # Dx_x = (displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1])
    # Dx_y = (displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1])

    # Normal grid
    if backward:
        D1 = (1 - Dx_x) * (1 - Dy_y)
    else:
        D1 = (1 + Dx_x) * (1 + Dy_y)
    # D1 = (Dx_x) * (Dy_y)
    D2 = Dx_y * Dy_x
    jacdet = D1 - D2

    # # tanh grid
    # grid_x = grid[:, 0, :-1, :-1]
    # grid_y = grid[:, 1, :-1, :-1]
    # coef = 1 - torch.tanh(torch.atanh(grid) + displacement)**2
    # coef_x = coef[:, 0, :-1, :-1]
    # coef_y = coef[:, 1, :-1, :-1]
    # D1 = (1 / (1 - grid_x**2) + Dx_x) * (1 / (1 - grid_y**2) + Dy_y)
    # D2 = Dx_y * Dy_x
    # jacdet = coef_x * coef_y * (D1 - D2)

    return jacdet


def jacdet_loss(vf, grid=None, backward=False):
    '''
    Penalizing locations where Jacobian has negative determinants
    Add to final loss
    '''
    # jacdet = get_jacdet2d(vf, grid, backward)
    # ans = 1 / 2 * (torch.abs(jacdet) - jacdet).mean(axis=[1, 2]).sum()
    # return ans

    jacdet = get_jacdet2d(vf, grid, backward)
    ans = 1 / 2 * (torch.abs(jacdet[jacdet > 1]).mean() + torch.abs(jacdet[jacdet < 0]).mean())
    if torch.isnan(ans):
        return 0
    return ans


def constrained_jacdet_loss(vf, grid=None, backward=False):
    '''
    regularization to ensure jac only range from 0-1
    '''
    jacdet = get_jacdet2d(vf, grid, backward)
    ans = 1 / 2 * (torch.abs(jacdet) - jacdet).mean(axis=[1, 2]).sum()
    return ans


def outgrid_loss(vf, grid, backward=False, size=32):  # 32-1):
    '''
    Penalizing locations where Jacobian has negative determinants
    Add to final loss
    '''
    if backward:
        pos = grid - vf - (size - 1)
        neg = grid - vf
    else:
        pos = grid + vf - (size - 1)
        neg = grid + vf

    # penalize > size
    ans_p = 1 / 2 * (torch.abs(pos) + pos).mean(axis=[1, 2]).sum()
    # penalize < 0
    ans_n = 1 / 2 * (torch.abs(neg) - neg).mean(axis=[1, 2]).sum()
    ans = ans_n + ans_p

    return ans
