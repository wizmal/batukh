from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


class MultipleRandomRotation(transforms.RandomRotation):
    def __init__(self,
                 degrees,
                 resample=False,
                 expand=False,
                 center=None,
                 fill=None):
        super(MultipleRandomRotation, self).__init__(
            degrees, resample, expand, center, fill)

    def __call__(self, images):

        if self.fill is None:
            self.fill = [None]*len(images)

        angle = self.get_params(self.degrees)
        return [TF.rotate(img, angle, self.resample, self.expand, self.center,
                          self.fill[i]) for i, img in enumerate(images)]


class MultipleColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, n_max=None):
        super(MultipleColorJitter, self).__init__(
            brightness, contrast, saturation, hue)

        self.n_max = n_max

    def __call__(self, images):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        if self.n_max is None:
            self.n_max = len(images)

        out = [transform(images[i]) for i in range(self.n_max)]
        out.extend(images[self.n_max:])
        return out


class MultipleToTensor(transforms.ToTensor):
    def __init__(self):
        super(MultipleToTensor, self).__init__()

    def __call__(self, images):
        return [TF.to_tensor(img) for img in images]
