import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyproj import Transformer
import matplotlib.pyplot as plt

from sentinelhub import (
    SentinelHubRequest,
    MosaickingOrder,
    DataCollection,
    MimeType,
    SHConfig,
    BBox,
    CRS,
    bbox_to_dimensions
)

from preprocessing import SatelliteDataset

STARTING_DATE = "2024-01-01"
ENDING_DATE = "2025-01-01"

CRS_WGS = "EPSG:4326"
CRS_UTM = "EPSG:32610"


def load_img_data(filepath_data, dirpath_save, half_size=1000, resolution=10, train=True):
    """half_size is in meters, and resolution is the number of metres per pixel."""

    lat, lon, id = get_lat_lon_id(filepath_data, train)

    coords = get_coords(lat, lon, half_size)

    config = env_load()

    evalscript = """
    // VERSION=3

    function setup()
    {
        return {
            input: ["B11", "B08", "B04", "B03"],
            output: {
                bands: 3,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(s)
    {
        let ndvi = index(s.B08, s.B04);
        let ndwi = index(s.B03, s.B08);
        let ndbi = index(s.B11, s.B08)
        return [ndvi, ndwi, ndbi];
    }
    """
    os.makedirs(dirpath_save, exist_ok=True)

    for i in range(id.shape[0]):

        if os.path.exists(os.path.join(dirpath_save, f"{id[i]}.npy")):
            continue

        bbox = BBox(bbox=tuple(coords[i]), crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(STARTING_DATE, ENDING_DATE),
                    mosaicking_order=MosaickingOrder.LEAST_CC
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=config
        )

        data = request.get_data()[0]
        np.save(os.path.join(dirpath_save, str(id[i])), data)

        print(f"Saved picture for id {id[i]}.")


def get_lat_lon_id(filepath, train):
    df = pd.read_excel(filepath)
    if train:
        _, _, id, lat, lon, _ = SatelliteDataset._preprocess(df, train)
    else:
        _, id, lat, lon, _ = SatelliteDataset._preprocess(df, train)
    print("Obtained latitudes, longitudes, and id.")
    return lat, lon, id


def get_coords(lat, lon, half_size):

    transformer_to_utm = Transformer.from_crs(CRS_WGS, CRS_UTM, always_xy=True)
    transformer_to_wgs = Transformer.from_crs(CRS_UTM, CRS_WGS, always_xy=True)

    x, y = transformer_to_utm.transform(lon, lat)

    min_x = x - half_size
    max_x = x + half_size
    min_y = y - half_size
    max_y = y + half_size

    min_lon, min_lat = transformer_to_wgs.transform(min_x, min_y)
    max_lon, max_lat = transformer_to_wgs.transform(max_x, max_y)

    coords = np.stack((min_lon, min_lat, max_lon, max_lat), axis=-1)

    print("Calculated coordinates.")

    return coords


def env_load():

    load_dotenv()

    config = SHConfig()
    config.sh_client_id = os.getenv("CLIENT_ID")
    config.sh_client_secret = os.getenv("CLIENT_SECRET")

    if not config.sh_client_id or not config.sh_client_secret:
        print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

    print("Acquired environment variables.")

    return config


def main():

    load_img_data("train.xlsx", "data", resolution=10, half_size=400)
    load_img_data("test.xlsx", "test", resolution=10, half_size=400, train=False)

    return



if __name__ == "__main__":
    main()
