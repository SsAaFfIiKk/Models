import torch
from torch import nn
from torch.utils.data import Dataset
from tools import *


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.batch1 = nn.BatchNorm2d(num_features=16)
        self.batch2 = nn.BatchNorm2d(num_features=32)
        self.batch3 = nn.BatchNorm2d(num_features=64)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.act = nn.ReLU()

        self.dropout = nn.Dropout(inplace=True)

        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.batch1(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv2(h)
        h = self.batch2(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv3(h)
        h = self.batch3(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv4(h)
        h = self.batch3(h)
        h = self.act(h)

        h = self.conv5(h)
        h = self.batch3(h)
        h = self.act(h)
        h = self.pool(h)

        h = h.view(h.size(0), -1)

        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc3(h)
        return h


class EyesDataset(torch.utils.data.Dataset):
    get_label = {'closed': 0, 'open': 1}

    def __init__(self, paths_to_images, transforms):
        self.paths_to_images = paths_to_images
        self.transforms = transforms

    def __len__(self):
        return len(self.paths_to_images)

    def __getitem__(self, idx):
        path_to_img = self.paths_to_images[idx]
        file_name = path_to_img.split('\\')[-1]
        img = Image.open(path_to_img)
        img_tensor = self.transforms(img)

        label_name = file_name[:-4].split('_')[-1]
        label_idx = self.get_label[label_name]

        return img_tensor, label_idx


train_los_list = []
test_los_list = []
train_acc_list = []
test_acc_list = []

weights_folder = "eye"
if os.path.exists(weights_folder):
    remove(weights_folder, "pth", "data")
else:
    os.mkdir(weights_folder)

test_transform, train_transform = get_transform((64, 64))

path_to_dataset = "F:/Python/Data/Faces"
paths_to_images = [os.path.join(path_to_dataset, name)
                   for name in os.listdir(path_to_dataset) if name.endswith('.jpg')]

random.seed(0)
random.shuffle(paths_to_images)

train_size = int(0.8 * len(paths_to_images))
batch_size = 256
epoch_num = 80

if __name__ == "__main__":
    train_dataset = EyesDataset(paths_to_images[:train_size], train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = EyesDataset(paths_to_images[train_size:], test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    cnn = CNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    cnn.to(device)

    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

    for epoch_idx in range(epoch_num):
        print('Epoch #{}'.format(epoch_idx))
        cnn.train()
        train_correct = 0
        train_loss = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = cnn(data)
            loss = error(pred, target)
            loss.backward()
            optimizer.step()

            _, pred_labels = torch.max(pred, dim=1)
            train_correct += (pred_labels == target).sum().item()
            train_loss += loss.item()

        train_los_list.append(train_loss / len(train_loader.dataset))
        train_acc_list.append(train_correct / len(train_loader.dataset))

        print('Train loss = {}'.format(train_loss / len(train_loader.dataset)))
        print('Train acccuracy = {}'.format(train_correct / len(train_loader.dataset)))
        print()

        cnn.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                pred = cnn(data)
                loss = error(pred, target)

                _, pred_labels = torch.max(pred, dim=1)
                test_correct += (pred_labels == target).sum().item()
                test_loss += loss.item()

            test_loss /= len(test_loader.dataset)
            test_correct /= len(test_loader.dataset)

            test_los_list.append(test_loss)
            test_acc_list.append(test_correct)

            print('Test loss = {}'.format(test_loss))
            print('Test acccuracy = {}'.format(test_correct))
            print('---------------------')

        print('Save')
        torch.save(cnn, weights_folder + "/" + weights_folder + "_current_" +
                   "epoch_{}, loss_{}, correct_{}".format(epoch_idx, test_loss, test_correct) + ".pth")

    save_list(weights_folder + "/" + weights_folder + "_train_los", train_los_list)
    save_list(weights_folder + "/" + weights_folder + "_train_acc", train_acc_list)
    save_list(weights_folder + "/" + weights_folder + "_test_los", test_los_list)
    save_list(weights_folder + "/" + weights_folder + "_test_acc", test_acc_list)
