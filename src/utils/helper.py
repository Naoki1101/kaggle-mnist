import os
import time
import random
import dataclasses
from contextlib import contextmanager

import numpy as np
import torch


@dataclasses.dataclass
class Timer:
    processing_time: int = 0

    @contextmanager
    def timer(self, name: str) -> None:
        t0 = time.time()
        yield
        t1 = time.time()
        processing_time = t1 - t0
        self.processing_time += round(processing_time, 2)
        if self.processing_time < 60:
            print(
                f"[{name}] done in {processing_time:.0f} s (Total: {self.processing_time:.2f} sec)"
            )
        elif self.processing_time < 3600:
            print(
                f"[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 60:.2f} min)"
            )
        else:
            print(
                f"[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 3600:.2f} hour)"
            )

    def get_processing_time(self) -> float:
        return round(self.processing_time / 60, 2)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
