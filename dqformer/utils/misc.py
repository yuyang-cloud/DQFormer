from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
import sys
from collections import OrderedDict

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def pad_stack(tensor_list: List[Tensor]):
    """
    pad each tensor on the input to the max value in shape[1] and
    concatenate them in a single tensor.
    Input:
        list of tensors [Ni,Pi]
    Output:
        tensor [sum(Ni),max(Pi)]
    """
    _max = max([t.shape[1] for t in tensor_list])
    batched = torch.cat([F.pad(t, (0, _max - t.shape[1])) for t in tensor_list])
    return batched


def sample_points(masks, masks_ids, n_pts, n_samples):
    # select n_pts per mask to focus on instances
    # plus random points up to n_samples
    # masks: bs,[N_th, Np]   masks_ids: bs,[N_th,[Np_ins]]
    sampled = []
    for ids, mm in zip(masks_ids, masks):
        if len(ids) != 0:
            m_idx = torch.cat(
                [
                    id[torch.randperm(n_pts)[:n_pts]] if id.shape[0] > n_pts else id
                    for id in ids
                ]
            )
        else:
            device = mm.device
            m_idx = torch.tensor([], device=device)
        r_idx = torch.randint(mm.shape[1], [n_samples - m_idx.shape[0]]).to(m_idx)
        idx = torch.cat((m_idx, r_idx))
        sampled.append(idx)
    return sampled


class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        # i = 0
        for module in self._modules.values():
            # print(i)
            input = module(input)
            # i += 1
        return input