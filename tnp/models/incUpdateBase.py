# Interface slass to outline methods needed to be implemented for incremental context update support.
from abc import ABC, abstractmethod
import torch
import torch.distributions as td

# This is the class for effecient incremental updates. It requires specification of knowledge of various known maximums.
# Incremental updating has been implemented for any size context, but this effecient version is just to make it even faster 
# for certain cases. Big O still the same and generic version tested. This is used for stuff like AR mode to make gains on small ctx
class IncUpdateEff(ABC):
    @abstractmethod
    def init_inc_structs(self, m: int):
        raise NotImplementedError
    
    @abstractmethod
    def update_ctx(self, xc: torch.Tensor, yc: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def query(self, xt: torch.Tensor, dy: int) -> td.Normal:
        raise NotImplementedError

