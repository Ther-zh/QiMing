"""与 torchvision.transforms.Compose 等价的轻量实现，避免 Jetson 上依赖 torchvision C++ 扩展。"""


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
