from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
# from federatedscope.model_heterogeneity.methods.FCCL.utils.conf import data_path
from PIL import Image
# from public_dataset import PublicDataset,random_loaders
from typing import Tuple
# from transforms.denormalization import DeNormalize

class MyCifar100(CIFAR100):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])  #先将图像转换成张量并将图像串联

        super(MyCifar100, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def load_Cifar100(pubaug, path):
    if pubaug == 'weak':
        selected_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])
    elif pubaug == 'strong':
        selected_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.RandomApply([
                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
             ], p=0.8),
             transforms.RandomGrayscale(p=0.2),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])
    train_dataset = MyCifar100(path, train=True, transform=selected_transform, download=True)
    return  train_dataset

# class PublicCIFAR100(PublicDataset):
#     NAME = 'pub_cifar100'
#
#     CON_TRANSFORM = transforms.Compose(
#         [transforms.RandomCrop(32, padding=4),
#          transforms.RandomHorizontalFlip(),
#          transforms.RandomApply([
#              transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#          ], p=0.8),
#          transforms.RandomGrayscale(p=0.2),
#          transforms.ToTensor(),
#          transforms.Normalize((0.4802, 0.4480, 0.3975),
#                               (0.2770, 0.2691, 0.2821))])
#
#     Nor_TRANSFORM = transforms.Compose(
#         [
#         transforms.RandomCrop(32, padding=4),
#          transforms.RandomHorizontalFlip(),
#          transforms.ToTensor(),
#          transforms.Normalize((0.4802, 0.4480, 0.3975),
#                               (0.2770, 0.2691, 0.2821))])
#
#     def get_data_loaders(self):
#         pub_aug = self.args.pub_aug
#         #选择transform
#         if pub_aug =='weak':
#             selected_transform = self.Nor_TRANSFORM
#         elif pub_aug =='strong':
#             selected_transform = self.CON_TRANSFORM
#         #训练的数据集
#         train_dataset = MyCifar100(data_path(), train=True,
#                                    download=True, transform=selected_transform)
#         traindl = random_loaders(train_dataset,self)  #随机采样
#         return traindl
#
#
#     @staticmethod
#     def get_normalization_transform():
#         transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
#                               (0.2770, 0.2691, 0.2821))
#         return transform
#
#     @staticmethod
#     def get_denormalization_transform():
#         transform = DeNormalize((0.4802, 0.4480, 0.3975),
#                               (0.2770, 0.2691, 0.2821))
#         return transform