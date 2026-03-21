from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr

class Schedule(BaseModel, ABC):
    start_value: float
    curr_step: int = 0
    _curr: float = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._curr = self.start_value

    @abstractmethod
    def step(self) -> None:
        self.curr_step += 1

    def get_value(self) -> float:
        return self._curr

class ConstantSchedule(Schedule):
    def step(self) -> None:
        super().step()
