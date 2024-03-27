from pathlib import Path
from typing import Optional

import pandas as pd
from lightning import pytorch as pl
from torch.utils.data import DataLoader

from datamodules.dataset_split import DatasetSplits
from datasets.ngram_data_set import NgramDataset
import numpy as np
class NgramDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: Path,
            batch_size: int = 32,
            num_workers: int = 4,
            train_size: float = 0.8,
            Context: int=2):
        super().__init__()

        self._data_path = Path(data_path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_size = train_size

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        #print(self._data_path)
        df = pd.read_csv(self._data_path)
        self._target_columns = ['steering_tire_angle',
                        'steering_tire_rotation_rate',
                        'acceleration',
                        'speed',
                        'jerk',
                        "pose_x","pose_y","pose_z","orientation_x","orientation_y","orientation_z","orientation_w",]
        self._feature_columns = sorted(list(set(df.columns).difference(set(self._target_columns))))
        df=df[self._feature_columns]
        colNr=len(df.columns)
        ColumnsPresets=["point_0_acceleration_mps2",
                    "point_0_front_wheel_angle_rad",
                    "point_0_heading_rate_rps",
                    "point_0_lateral_velocity_mps",
                    "point_0_longitudinal_velocity_mps",
                    "point_0_rear_wheel_angle_rad",
                    "point_0_pos_x","point_0_pos_y",
                    "point_0_pos_z","point_0_orientation_x",
                    "point_0_orientation_y",
                    "point_0_orientation_z",
                    "point_0_orientation_w"]
        columns=[]
        sufixes=[]
        #print(df.columns)
        for i in range(Context):
            sufixes.append("N-"+str(Context-i))
        sufixes.reverse()
        sufixes.append("N")
        #print(sufixes)
        for sufix in sufixes:
            for column in ColumnsPresets:
                columns.append(column.replace("0",sufix))
        #print(columns)
        data=pd.DataFrame(columns=columns)
        for i in range(len(df.index)):
            for j in range(6):
                newrow=np.array([])
                for k in range(Context+1):
                    low=j*len(ColumnsPresets)+len(ColumnsPresets)*k
                    high=j*len(ColumnsPresets)+len(ColumnsPresets)*(k+1)
                    #print(len(df.iloc[i][low:high]))
                    newrow=np.append([newrow],[df.iloc[i][low:high]])
                data=data.append(pd.DataFrame(newrow.reshape(1,-1), columns=list(data)), ignore_index=False)
#        print(data)
        train_df, val_df, test_df = DatasetSplits.basic_split(data, self._train_size)
        print(len(train_df))
        print(len(test_df))
        self.train_dataset = NgramDataset(
            train_df,Context,len(ColumnsPresets)
        )

        self.train_dataset = NgramDataset(
            val_df,Context,len(ColumnsPresets)
        )

        self.test_dataset = NgramDataset(
            test_df,Context,len(ColumnsPresets)
        )

        self.save_hyperparameters(ignore=['data_path', 'number_of_workers'])

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )
