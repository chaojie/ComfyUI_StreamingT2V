# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.

import random

import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
