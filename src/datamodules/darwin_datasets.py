import os
from typing import Any, Dict, Optional, Tuple, TypeAlias

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Affine
from torch.utils.data import Dataset

MaskRCNNInput: TypeAlias = Tuple[Any, Dict[str, Any], Tuple[int, str]]
ClassifierInput: TypeAlias = Tuple[Any, torch.Tensor, Tuple[int, str]]


class ClassifierDataset(Dataset):
    def __init__(self, csv: str, root_dir: str, transforms: Optional[A.Compose] = None, **kwargs) -> None:
        data = pd.read_csv(csv)

        data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: os.path.join(root_dir, x))

        file_exists = data.apply(lambda row: cv2.haveImageReader(row.iloc[0]), axis=1)

        data = data[file_exists]

        self.images = data.iloc[:, 0].tolist()
        self.labels = data.iloc[:, 1].tolist()

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> ClassifierInput:
        image_path = self.images[idx]
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.transforms:
            image = self.transforms(image=image, label=label)["image"]

        return image, label, (idx, image_name)


class MaskRCNNDataset(Dataset):
    def __init__(self, csv: str, root_dir: str, transforms: Optional[A.Compose] = None, **kwargs) -> None:
        data = pd.read_csv(csv)

        data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: os.path.join(root_dir, x))
        data.iloc[:, 1] = data.iloc[:, 1].apply(lambda x: os.path.join(root_dir, x))

        file_exists = data.apply(
            lambda row: cv2.haveImageReader(row.iloc[0]) and cv2.haveImageReader(row.iloc[1]), axis=1
        )

        data = data[file_exists]

        self.images = data.iloc[:, 0].tolist()
        self.labels = data.iloc[:, 1].tolist()
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> MaskRCNNInput:
        image_path = self.images[idx]
        image_name = os.path.basename(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mask_path = self.labels[idx]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            boxes.append(
                [
                    np.min(pos[1]),
                    np.min(pos[0]),
                    np.max(pos[1]),
                    np.max(pos[0]),
                ]
            )

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = [np.array(mask, dtype=np.uint8) for mask in masks]

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            sample = {
                "image": image,
                "bboxes": boxes,
                "masks": masks,
                "labels": labels,
            }

            sample = self.transforms(**sample)

            image = sample["image"]
            target["boxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            target["masks"] = torch.as_tensor(np.stack(sample["masks"], axis=0), dtype=torch.uint8)

        return image, target, (idx, image_name)
