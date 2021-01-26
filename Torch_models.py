import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms
from Tools import *


class EmotDataset(torch.utils.data.Dataset):
    # Порядок такой же как и у Артёма
    get_label = {"angry": 0,
                 "happy": 1,
                 "neutral": 2,
                 "sad": 3,
                 "fear": 4}

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

num_classes = 5
batch_size = 256
epoch_num = 200
feature_extract = True

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

weights_folder = "res50"
if os.path.exists(weights_folder):
    remove(weights_folder, ".pth", ".data")
else:
    os.mkdir(weights_folder)

if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomApply([MotionBlur(),
                                transforms.GaussianBlur((5, 5), sigma=(0.1, 2.0))],
                               p=0.3),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0),
        transforms.RandomCrop(64, padding=6),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    path_to_dataset = "F:/Python/Data/Emot"
    paths_to_images = [os.path.join(path_to_dataset, name)
                       for name in os.listdir(path_to_dataset) if name.endswith('.jpg')]

    random.seed(0)
    random.shuffle(paths_to_images)

    train_size = int(0.8 * len(paths_to_images))

    train_dataset = EmotDataset(paths_to_images[:train_size], train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = EmotDataset(paths_to_images[train_size:], test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch_idx in range(epoch_num):
        print('Epoch #{}'.format(epoch_idx))
        model.train()
        train_correct = 0
        train_loss = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(data)
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

        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                pred = model(data)
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

            torch.save(model, weights_folder + "/" + weights_folder + "_current_" +
                       "epoch_{}, loss_{}, correct_{}".format(epoch_idx, test_loss, test_correct) + ".pth")

    save_list(weights_folder + "/" + weights_folder + "_train_los", train_los_list)
    save_list(weights_folder + "/" + weights_folder + "res_train_acc", train_acc_list)
    save_list(weights_folder + "/" + weights_folder + "res_test_los", test_los_list)
    save_list(weights_folder + "/" + weights_folder + "res_test_acc", test_acc_list)
