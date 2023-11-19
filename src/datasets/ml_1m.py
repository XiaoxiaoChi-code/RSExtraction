from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def is_7zfile(cls):
        return False

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and \
                all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return

        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.zip')
        tmpfolder = tmproot.joinpath('folder')
        download(self.url(), tmpzip)
        unzip(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print()

    def preprocess(self):
        # 我要找到相应的路径，然后把我的数据文件放在这个路径下，让程序直接去读取这个数据文件，而不是重新下载
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        # 注释掉下载原始数据集的代码
        # self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        # 到这一步之前都是读取数据，得到了dataframe 类型的数据

        # 按照条件 过滤掉不满足要求的用户和商品， 得到过滤之后的df
        df = self.filter_triplets(df)

        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):

        # folder_path = self._get_rawdata_folder_path()
        # file_path = folder_path.joinpath('ratings.dat')

        file_path1 = "/content/gdrive/MyDrive/practice/RSExtraction/src/datasets/MovieLens_target_member_dataset.csv"
        df1 = pd.read_csv(file_path1, sep=',')

        file_path2 = "/content/gdrive/MyDrive/practice/RSExtraction/src/datasets/MovieLens_target_nonMember_dataset.csv"
        df2 = pd.read_csv(file_path2, sep=',')

        df = pd.concat([df1,df2], ignore_index=True, sort=False)

        # df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
