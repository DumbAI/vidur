from vidur.scheduler.global_scheduler.base_scheduler import BaseScheduler

class APIScheduler(BaseScheduler):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas

    def schedule(self) -> List[Tuple[int, Request]]:
        

        
