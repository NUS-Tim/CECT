import torch.utils.data as tud
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


def data_aug(dataset, bs, device):

    train_dir = './datasets/' + str(dataset) + '/training'
    validation_dir = './datasets/' + str(dataset) + '/validation'
    test_dir = './datasets/' + str(dataset) + '/test'

    t_tra = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    vt_tra = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    train_d = ImageFolder(train_dir, transform=t_tra)
    val_d = ImageFolder(validation_dir, transform=vt_tra)
    test_d = ImageFolder(test_dir, transform=vt_tra)

    if str(device) == 'cuda':
        train_l = tud.DataLoader(train_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
        val_l = tud.DataLoader(val_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
        test_l = tud.DataLoader(test_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        train_l = tud.DataLoader(train_d, batch_size=bs, shuffle=True)
        val_l = tud.DataLoader(val_d, batch_size=bs, shuffle=True)
        test_l = tud.DataLoader(test_d, batch_size=bs, shuffle=True)

    return train_l, val_l, test_l, train_d, val_d, test_d
