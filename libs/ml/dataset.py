from collections import namedtuple
import torch


PottsSample = namedtuple("PottsSample", ["spins", "label"])


class PottsDatasetBase(torch.utils.data.Dataset):
    def __getitem__(self, index: int) -> PottsSample:
        raise NotImplementedError()


class GroundStateDataset(PottsDatasetBase):
    def __init__(self, Q: int, Lx: int, Ly: int, repeat: int):
        super().__init__()
        self.Q = Q
        self.Lx = Lx
        self.Ly = Ly
        self.repeat = repeat

    def __len__(self) -> int:
        return self.Q * self.repeat
    
    def _get_sample(self, i: int) -> PottsSample:
        spins = torch.zeros((self.Q, self.Lx, self.Ly))
        spins[i, ...] = 1
        label = torch.tensor(i, dtype=torch.long)
        return PottsSample(spins, label)
        
    def __getitem__(self, index: int) -> PottsSample:
        assert 0 <= index < len(self)
        return self._get_sample(index % self.Q)
