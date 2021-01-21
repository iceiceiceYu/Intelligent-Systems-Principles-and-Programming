import torchvision
import torchvision.datasets as datasets


def get_data(path):
    dataset = datasets.ImageFolder(root=path, transform=torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.ToTensor(),
    ]))
    return dataset
