from typing import Tuple, TypeAlias, Literal

class SplitConfig:
    def __init__(
        self,
        split: Tuple[float, float, float],
        seed: int,
        use_stratified: bool,
    ) -> None:
        self.split = split
        self.seed = seed
        self.use_stratified = use_stratified

Mode: TypeAlias = Literal["train", "test", "val"]
