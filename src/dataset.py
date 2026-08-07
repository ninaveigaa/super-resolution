"""
Data pipeline in PyTorch.

Equivalent to:
- createTrainingCubes2(args, HR, LRxy, batchsize, cropsize, scale) -> create_training_cubes
- augmentData(image)                                               -> augment_data

Keeps the same sampling logic (cycles through the xy/yz/xz planes to extract
"cubes" from a preloaded HR volume and LR volume stored as .npy), but returns
tensors already in NCHW (PyTorch) instead of NHWC (TF).
"""

import sys
import numpy as np
import torch


def create_training_cubes(args, HR, LRxy, batchsize, cropsize, scale):
    """
    Extracts LR/HR block batches from the preloaded volume, cycling through
    the 3 planes (xy, yz, xz) each iteration, same as the original script.

    Returns float32 numpy arrays in [-1, 1], still in (N, H, W, 1) layout
    (conversion to NCHW is done in the EpochCubes wrapper below, when turning
    them into tensors).
    """
    batchLR = np.zeros([batchsize * args.itersPerEpoch, cropsize, cropsize, 1], 'float32')
    batchHR = np.zeros([batchsize * args.itersPerEpoch * scale, cropsize * scale, cropsize * scale, 1], 'float32')
    n = 0
    n2 = 0
    for i in range(args.itersPerEpoch):
        if np.mod(i, 3) == 0:
            x = int(np.floor(np.random.rand() * (LRxy.shape[0] - batchsize)))
            y = int(np.floor(np.random.rand() * (LRxy.shape[1] - cropsize)))
            z = int(np.floor(np.random.rand() * (LRxy.shape[2] - cropsize)))

            block = np.expand_dims(LRxy[x:x + batchsize, y:y + cropsize, z:z + cropsize], 3)
            blockHR = np.expand_dims(
                HR[x * scale:x * scale + batchsize * scale,
                   y * scale:y * scale + cropsize * scale,
                   z * scale:z * scale + cropsize * scale], 3)

        elif np.mod(i, 3) == 1:
            x = int(np.floor(np.random.rand() * (LRxy.shape[0] - cropsize)))
            y = int(np.floor(np.random.rand() * (LRxy.shape[1] - cropsize)))
            z = int(np.floor(np.random.rand() * (LRxy.shape[2] - batchsize)))

            block = np.expand_dims(LRxy[x:x + cropsize, y:y + cropsize, z:z + batchsize], 3)
            blockHR = np.expand_dims(
                HR[x * scale:x * scale + cropsize * scale,
                   y * scale:y * scale + cropsize * scale,
                   z * scale:z * scale + batchsize * scale], 3)
            block = np.transpose(block, [2, 0, 1, 3])
            blockHR = np.transpose(blockHR, [2, 0, 1, 3])

        else:  # np.mod(i, 3) == 2
            x = int(np.floor(np.random.rand() * (LRxy.shape[0] - cropsize)))
            y = int(np.floor(np.random.rand() * (LRxy.shape[1] - batchsize)))
            z = int(np.floor(np.random.rand() * (LRxy.shape[2] - cropsize)))

            block = np.expand_dims(LRxy[x:x + cropsize, y:y + batchsize, z:z + cropsize], 3)
            blockHR = np.expand_dims(
                HR[x * scale:x * scale + cropsize * scale,
                   y * scale:y * scale + batchsize * scale,
                   z * scale:z * scale + cropsize * scale], 3)
            block = np.transpose(block, [1, 0, 2, 3])
            blockHR = np.transpose(blockHR, [1, 0, 2, 3])

        batchLR[n:n + batchsize] = block / 127.5 - 1
        batchHR[n2:n2 + batchsize * scale] = blockHR / 127.5 - 1
        n += batchsize
        n2 += batchsize * scale

        sys.stdout.write("\rHR Cube: %d of %d" % (i + 1, args.itersPerEpoch))
        sys.stdout.flush()
    sys.stdout.write("\n")
    return batchHR, batchLR


def augment_data(image):
    """
    Equivalent to augmentData: random contrast/brightness + clip.
    `image` is a torch tensor in [-1, 1], NCHW layout.
    """
    cont_factor = (np.random.rand() * 2 - 1) * 0.2 + 1
    bright_factor = (np.random.rand() * 2 - 1) * 0.2 + 1

    image = image * bright_factor
    image = (image - image.mean()) * cont_factor + image.mean()
    image = torch.clamp(image, -1, 1)
    return image


class EpochCubes:
    """
    Holds a single epoch's large arrays (already in NCHW) and knows how to
    slice them into minibatches of size `batch_size` (LR) / `batch_size*scale`
    (HR), exactly like HR_dataset.batch(batchsize*scale) / LR_dataset.batch(batchsize)
    did in the original tf.data script.
    """

    def __init__(self, batchHR, batchLR, batch_size, scale, device):
        # (N, H, W, 1) -> (N, 1, H, W), as tensors already on the training device
        self.HR = torch.from_numpy(np.transpose(batchHR, [0, 3, 1, 2])).to(device)
        self.LR = torch.from_numpy(np.transpose(batchLR, [0, 3, 1, 2])).to(device)
        self.batch_size = batch_size
        self.scale = scale
        self.num_chunks = self.LR.shape[0] // batch_size

    def iter_batches(self):
        """One pass over the available chunks (equivalent to `for x, y in zip(HR_dataset, LR_dataset)`)."""
        bs = self.batch_size
        for i in range(self.num_chunks):
            hr = self.HR[i * bs * self.scale: (i + 1) * bs * self.scale]
            lr = self.LR[i * bs: (i + 1) * bs]
            yield hr, lr

    def val_batches(self, val_num):
        """Equivalent to HR_dataset_test / LR_dataset_test (first `val_num` chunks)."""
        bs = self.batch_size
        for i in range(min(val_num, self.num_chunks)):
            hr = self.HR[i * bs * self.scale: (i + 1) * bs * self.scale]
            lr = self.LR[i * bs: (i + 1) * bs]
            yield hr, lr


def make_epoch_cubes(args, HR, LRxy, batch_size, crop_size, scale, device):
    """Equivalent to calling createTrainingCubes2 and building the tf.data.Dataset objects."""
    batchHR, batchLR = create_training_cubes(args, HR, LRxy, batch_size, crop_size, scale)
    return EpochCubes(batchHR, batchLR, batch_size, scale, device)
