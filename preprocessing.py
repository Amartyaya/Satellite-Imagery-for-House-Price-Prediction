import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset


class SatelliteDataset(Dataset):

    def __init__(self, df, train=True, img=False, train_img_path=None, test_img_path=None):

        self.train = train
        self.img = img

        self.train_img_path = train_img_path
        self.test_img_path = test_img_path

        if self.train:
            X, y, id, _, _, zip = self._preprocess(df, self.train)
        else:
            X, id, _, _, zip = self._preprocess(df, self.train)

        self.X = torch.from_numpy(X.copy())
        self.id = id
        self.zip = torch.from_numpy(zip.copy())
        if self.train:
            self.y = torch.from_numpy(y.copy()).unsqueeze(-1)

    @staticmethod
    def _preprocess(df, train):

        id = df.id.values
        zipcode = df.zipcode.astype("category").cat.codes.values.astype("long")
        lat = df.lat.values
        long = df.long.values
        if train:
            y = np.log(df.price.values).astype(np.float32)

        df.sqft_living = np.log(df.sqft_living.values)
        df.sqft_lot = np.log(df.sqft_lot.values)
        df.sqft_living15 = np.log(df.sqft_living15.values)
        df.sqft_lot15 = np.log(df.sqft_lot15.values)

        sale_year = df.date.apply(lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S").year).values
        effective_age = sale_year - df.yr_renovated.where(df.yr_renovated != 0, df.yr_built)
        effective_age[effective_age < 0] = 0
        is_renovated = df.yr_renovated.where(df.yr_renovated == 0, 1)

        df = df.assign(effective_age=effective_age, is_renovated=is_renovated)

        if train:
            X = df.drop(columns=["id", "date", "price", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long"], axis=1)
        else:
            X = df.drop(columns=["id", "date", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long"], axis=1)
        X = X.values.astype(np.float32)

        if train:
            return X, y, id, lat, long, zipcode
        else:
            return X, id, lat, long, zipcode

    def __getitem__(self, idx):

        if self.train:
            if self.img:
                image = np.load(os.path.join(self.train_img_path, f"{self.id[idx]}.npy"))
                image_tensor = torch.from_numpy(image.copy().astype(np.float32)).permute(2, 0, 1)
                return self.X[idx], self.y[idx], self.zip[idx], image_tensor
            else:
                return self.X[idx], self.y[idx], self.zip[idx]
        
        else:
            if self.img:
                image = np.load(os.path.join(self.test_img_path, f"{self.id[idx]}.npy"))
                image_tensor = torch.from_numpy(image.copy().astype(np.float32)).permute(2, 0, 1)
                return self.X[idx], self.zip[idx], image_tensor
            else:
                return self.X[idx], self.zip[idx]

    def __len__(self):
        return self.X.shape[0]
