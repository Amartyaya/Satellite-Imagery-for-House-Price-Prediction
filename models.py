import torch
import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, emb_dim):

        super().__init__()

        self.dense_layer1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim)
        )

        self.emb_layer = nn.Sequential(
            nn.Embedding(emb_dim, hid_dim),
            nn.Linear(hid_dim, hid_dim)
        )

        self.dense_layer2 = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim * 2),
            nn.Linear(hid_dim * 2, out_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.uniform_(m.bias)

    def forward(self, x, zip):
        fuse = torch.cat([self.dense_layer1(x), self.emb_layer(zip)], dim=-1)
        return self.dense_layer2(fuse)


class CNet(nn.Module):

    def __init__(self, img_dim, in_channels, hid_dim, num_groups):

        assert in_channels % num_groups == 0, f"in_channels {in_channels} must be divisible by num_groups {num_groups}."

        super().__init__()

        hid_channels_1 = in_channels + num_groups
        hid_channels_2 = in_channels + 2 * num_groups
        hid_channels_3 = in_channels + 3 * num_groups

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels_1, kernel_size=2, stride=2, groups=num_groups, bias=False),
            nn.Conv2d(hid_channels_1, hid_channels_1, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hid_channels_1),
            nn.Conv2d(hid_channels_1, hid_channels_2, kernel_size=2, stride=2, groups=num_groups, bias=False),
            nn.Conv2d(hid_channels_2, hid_channels_2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hid_channels_2),
            nn.Conv2d(hid_channels_2, hid_channels_3, kernel_size=2, stride=2, groups=num_groups, bias=False),
            nn.Conv2d(hid_channels_3, hid_channels_3, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hid_channels_3),
            nn.Flatten()
        )

        new_img_dim = img_dim // 8

        self.linear_layer = nn.Linear(hid_channels_3 * new_img_dim * new_img_dim, hid_dim)

    def forward(self, x):
        return self.linear_layer(self.conv_layer(x))


class Net(nn.Module):

    def __init__(self, in_dim, img_dim, hid_dim, out_dim, in_channels, emb_dim, num_groups):

        super().__init__()

        self.dense_layer1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim)
        )

        self.emb_layer = nn.Sequential(
            nn.Embedding(emb_dim, hid_dim),
            nn.Linear(hid_dim, hid_dim)
        )
        
        self.conv_layer = CNet(img_dim, in_channels, hid_dim, num_groups)

        self.dense_layer2 = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim * 3),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim * 3),
            nn.Linear(hid_dim * 3, out_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.uniform_(m.bias)
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)

    def forward(self, x, zip, img):
        fuse = torch.cat([self.dense_layer1(x), self.emb_layer(zip), self.conv_layer(img)], dim=1)
        return self.dense_layer2(fuse)


def main():

    model = Net(14, 80, 20, 1, 3, 70, 3)
    # print(model)
    print(sum([param.numel() for param in model.parameters()]))
    X = torch.ones(2, 14)
    zip = torch.ones(2).to(torch.int)
    image = torch.ones((2, 3, 80, 80))

    pred = model(X, zip, image)
    print(pred.shape)

    mlp = MLP(14, 20, 1, 70)
    print(sum([param.numel() for param in mlp.parameters()]))

    pred_mlp = mlp(X, zip)
    print(pred_mlp.shape)

    return



if __name__ == "__main__":
    main()
