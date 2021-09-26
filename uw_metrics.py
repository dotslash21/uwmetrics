import os
import re
from typing import List, Tuple, Dict

import numpy
from PIL import Image
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from nmetrics import nmetrics


class UwMetrics:
    def __init__(
            self,
            original_dir: str = "data/original/",
            restored_dir: str = "data/restored/",
            original_prefix: str = "real_A_103_",
            restored_prefix: str = "funiegan_output_"):

        self.original_dir = original_dir
        self.restored_dir = restored_dir
        self.original_prefix = original_prefix
        self.restored_prefix = restored_prefix

    def extract_index(self, original_image_filename: str):
        regex = r"^{}([0-9]+)\.jpg$".format(self.original_prefix)
        return int(re.match(regex, original_image_filename).groups()[0])

    def load_images(self) -> List[Tuple[int, numpy.ndarray, numpy.ndarray]]:
        image_pairs = list()

        for original_image_filename in os.listdir(self.original_dir):
            index = self.extract_index(original_image_filename)

            original_image = numpy.asarray(Image.open(os.path.join(self.original_dir, original_image_filename)))

            restored_image_filename = original_image_filename.replace(self.original_prefix, self.restored_prefix)
            restored_image = numpy.asarray(Image.open(os.path.join(self.restored_dir, restored_image_filename)))

            image_pairs.append((index, original_image, restored_image))

        return image_pairs

    def calculate(self) -> List[Dict[str, float]]:
        image_pairs = self.load_images()

        metrics_list = list()

        for image_pair in image_pairs:
            index = image_pair[0]
            original_image = image_pair[1]
            restored_image = image_pair[2]

            psnr = peak_signal_noise_ratio(original_image, restored_image)
            ssim = structural_similarity(original_image, restored_image, multichannel=True)
            entropy_underwater = shannon_entropy(original_image)
            entropy_restored = shannon_entropy(restored_image)
            uiqm_restored, uciqe_restored = nmetrics(restored_image)
            uiqm_underwater, uciqe_underwater = nmetrics(original_image)

            metrics_list.append({
                "index": index,
                "psnr": psnr,
                "ssim": ssim,
                "entropy_underwater": entropy_underwater,
                "entropy_restored": entropy_restored,
                "uiqm_underwater": uiqm_underwater,
                "uiqm_restored": uiqm_restored,
                "uciqe_underwater": uciqe_underwater,
                "uciqe_restored": uciqe_restored
            })

        return metrics_list
