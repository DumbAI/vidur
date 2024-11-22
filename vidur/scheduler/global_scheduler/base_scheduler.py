from abc import ABC, abstractmethod
from typing import List, Tuple

from vidur.entities import Request


class BaseScheduler(ABC):

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass