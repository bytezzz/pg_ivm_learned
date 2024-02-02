import numpy as np
from typing import List, Tuple
import socket
from ctypes import *
import time

class MyStruct(Structure):
    _pack_ = 1

class RequestEmbed(MyStruct):
    _fields_ = [("embedding", c_double * 1024)]

class ScheduleDecision(MyStruct):
    _fields_ = [("decision", c_uint32)]

class EnvFeatures(MyStruct):
    _fields_ = [("lru", c_int * 17)]

@classmethod
def recv_and_unpack(cls, socket: socket.socket):
  buff = socket.recv(sizeof(cls))
  return cls.from_buffer_copy(buff)

MyStruct.recv_and_unpack = recv_and_unpack

class Reqs:
  def __init__(self,
               requests_embedding: np.ndarray,
               environment_embedding: np.ndarray,
               socket: socket.socket
               ) -> None:
    self.requests_embedding = requests_embedding
    self.environment_embedding = environment_embedding
    self.socket = socket

  def get_random_action(self) -> np.uint32:
    return np.random.randint(0, self.requests_embedding.shape[0], dtype=np.uint32)

  def get_env_features(self) -> np.ndarray:
    return self.environment_embedding

  def get_action_features(self) -> np.ndarray:
    """
    :return: (max_actions, action_features_dim)
    """
    return self.requests_embedding

  def make_decision(self, actions: np.uint32) -> None:
    response = ScheduleDecision()
    response.decision = actions

    self.socket.send(response)


class Engine:
  def __init__(self) -> None:
    self.ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.fired = False
    print("Socket created")

  def bind(self, host, port):
    self.ssock.bind((host, port))
    self.ssock.listen(3)
    print("Server listening on port {:d}".format(port))

    self.csock, self.client_address = self.ssock.accept()
    print("Accepted connection from {:s}".format(self.client_address[0]))

  def fetch_req(self) -> Reqs:

    tensor_lists = []

    env_features = np.array(EnvFeatures.recv_and_unpack(self.csock).lru, dtype=np.float32)

    #print(f"Received LRU: {np.array(env_features.lru)}")

    while payload_in := RequestEmbed.recv_and_unpack(self.csock):

      embedding = np.array(payload_in.embedding)
      if np.isnan(embedding).all():
        break

      #Set invalid embedding to 0
      embedding[np.isnan(embedding)] = 0.0

      tensor_lists.append(embedding)

    #print("Received {:d} tensors".format(len(tensor_lists)))

    if not self.fired:
      self.fired = True
      self.prev_time = time.time()
      return Reqs(np.array(tensor_lists), env_features, self.csock)

    reward = -(time.time() - self.prev_time) * len(tensor_lists)
    self.prev_time = time.time()
    return reward, Reqs(np.array(tensor_lists), env_features, self.csock)


  def close(self):
    self.ssock.close()
    self.csock.close()

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
    engine.bind("localhost", 2300)
    reqs = engine.fetch_req()
    while(True):
      reqs.make_decision(reqs.get_random_action())
      reward, reqs = engine.fetch_req()
      print(reward)
  except:
    print("Error")
    engine.ssock.close()
    raise
