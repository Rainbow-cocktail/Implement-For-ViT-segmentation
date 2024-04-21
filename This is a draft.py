import copy
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import SegmentationDataset
import torch
from nets import My_model
import torch
import PIL.Image as Image

a = torch.load(r'D:\python_project\My_ViT_SegNet\weights\exp6\best_model.pth')
