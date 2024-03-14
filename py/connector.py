import numpy as np
from typing import List, Tuple
from ctypes import *
import time
from sequence_generator import SequenceGenerator
from typing import NamedTuple
from enum import Enum
import zmq
import json


class ScheduleTag(Enum):
    QUERY_FINISHED = 0
    QUERY_GIVEUP = 1
    INCOMING_QUERY = 2
    DEAD_WAKEUP = 3

class EvaluationResult(NamedTuple):
    reward: float
    schedule_tag: ScheduleTag
    wakeup_by: int


class Reqs:
    def __init__(
        self,
        plans: list,
        environment_embedding: np.ndarray,
        send_hook
    ) -> None:
        self.environment_embedding = environment_embedding
        self.plans = plans
        self.send_hook = send_hook

    def get_random_action(self) -> np.uint32:
        return np.random.randint(0, len(self.plans), dtype=np.uint32)

    def get_env_features(self) -> np.ndarray:
        return self.environment_embedding

    def get_action_features(self) -> np.ndarray:
        """
        :return: (max_actions, action_features_dim)
        """
        return self.requests_embedding

    def make_decision(self, actions: np.uint32) -> int:
        response = {}
        response["decision"] = int(actions)
        response["decision_id"] = SequenceGenerator.get()
        if actions > (len(self.plans) - 1):
            print("ERROR", actions, len(self.plans))

        self.send_hook(json.dumps(response))

        return response["decision_id"]



class Engine:
    def __init__(self) -> None:
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.REP)
        self.sock.bind("tcp://*:5555")
        self.fired = False
        self.drop_next_reward = False
        self.running_mean = 0.0
        print("Socket created")

    def fetch_req(self) -> Tuple[EvaluationResult, Reqs]:
        tensor_lists = []

        request = json.loads(self.sock.recv())

        print(request)

        #env_features = EnvFeatures.recv_and_unpack(self.csock)
        env_features = request["env"]

        env_embedding = np.array(
            env_features["LRU"], dtype=np.float32
        )

        candidate_plans = request["candidate_plans"]

        #print("Received {:d} tensors".format(len(tensor_lists)))

        if not self.fired:
            self.fired = True
            self.prev_time = time.time()
            return Reqs(candidate_plans, env_embedding, lambda x: self.sock.send_string(x))

        reward = -(time.time() - self.prev_time) * len(tensor_lists)
        self.running_mean = (
            reward
            if self.running_mean == 0
            else 0.75 * self.running_mean + 0.25 * reward
        )
        self.prev_time = time.time()
        return EvaluationResult(reward, ScheduleTag(env_features["schedule_tag"]), env_features["wakeup_decision_id"]), Reqs(candidate_plans, env_embedding, lambda x: self.sock.send_string(x))


class ParallelEngines:
    def __init__(self) -> None:
        pass

    def connect(self, *args, **kwargs):
        pass

    def fetch_req(self) -> List[Reqs]:
        pass


if __name__ == "__main__":
    try:
        engine = Engine()
        #engine.bind("localhost", 2300)
        reqs = engine.fetch_req()
        while True:
            reqs.make_decision(reqs.get_random_action())
            evaluation_result, reqs = engine.fetch_req()
            print(evaluation_result)
    except:
        print("Error")
        raise
