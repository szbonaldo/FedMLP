import torch
from torchvision.transforms import transforms
# ChestXray14
from dataset.all_dataset import ChestXray14


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = ChestXray14('/home/szb/multilabel', "train", transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))
    print(getStat(train_dataset))
