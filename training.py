import torch
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from models import MLP, Net
from preprocessing import SatelliteDataset


def train_one_epoch(model, loader, optimizer, device, img):

    total_loss = 0.0
    total_samples = 0.0

    model.train()

    for batch in loader:
        
        if img:
            X, y, zip, image = batch
            image = image.to(device)
        else:
            X, y, zip = batch

        X = X.to(device)
        y = y.to(device)
        zip = zip.to(device)

        optimizer.zero_grad()

        pred = model(X, zip, image) if img else model(X, zip)
        loss = F.mse_loss(pred, y, reduction="sum")
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_samples += y.shape[0]

    return total_loss / total_samples


@torch.inference_mode
def validate_one_epoch(model, loader, device, img):

    total_loss = 0.0
    total_samples = 0.0

    model.eval()

    for batch in loader:

        if img:
            X, y, zip, image = batch
            image = image.to(device)
        else:
            X, y, zip = batch

        X = X.to(device)
        y = y.to(device)
        zip = zip.to(device)

        pred = model(X, zip, image) if img else model(X, zip)
        loss = F.mse_loss(pred, y, reduction="sum")
        total_loss += loss.item()
        total_samples += y.shape[0]

    return total_loss / total_samples


def train(model, dataset, optimizer, split, batch_size, epochs, num_workers, device, img=False):

    train_loss_list = []
    valid_loss_list = []

    train_size = int(split * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for epoch in range(epochs):
        
        print(f"\nEpoch {epoch + 1}:")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, img)
        train_loss_list.append(train_loss)
        print(f"Training loss: {train_loss}")

        valid_loss = validate_one_epoch(model, valid_loader, device, img)
        valid_loss_list.append(valid_loss)
        print(f"Validation loss: {valid_loss}")

    return train_loss_list, valid_loss_list


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_excel("train.xlsx")
    # dataset = SatelliteDataset(df)

    # model = MLP(14, 8, 1, 70)
    # model = model.to(device)
    #
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
    #
    # train_loss, valid_loss = train(model, dataset, optimizer, 0.7, 64, 10, 2, device)
    #
    # x = [i for i in range(1, len(train_loss)+1)]
    #
    # plt.plot(x, train_loss, 'r', x, valid_loss, 'g')
    # plt.show()

    dataset = SatelliteDataset(df, img=True, train_img_path=".\\data")

    model = Net(14, 80, 8, 1, 3, 70, 3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
    train_loss, valid_loss = train(model, dataset, optimizer, 0.7, 64, 10, 2, device, img=True)

    x = [i for i in range(1, len(train_loss)+1)]

    plt.plot(x, train_loss, 'r', x, valid_loss, 'g')
    plt.show()

    return



if __name__ == "__main__":
    main()
