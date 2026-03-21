from abc import ABC, abstractmethod
from pydantic import BaseModel

class Schedule(ABC, BaseModel):
    _curr: float
    _step: int = 0

    @abstractmethod
    def step(self):
        self._step += 1
        pass

    def get_value(self) -> float:
        return self._curr

class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        self._curr = value

    def step(self):
        super().step()
