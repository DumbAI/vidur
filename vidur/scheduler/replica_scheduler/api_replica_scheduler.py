from vidur.entities.batch import Batch
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)

# One session of API request
class APIReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

    def _get_next_batch(self) -> Batch:
        
        requests = []
        num_tokens = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)