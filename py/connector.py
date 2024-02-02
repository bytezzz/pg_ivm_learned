import numpy as np
from typing import List, Tuple
import socket
from ctypes import *
import time

class RequestEmbed(Structure):
    _pack_ = 1
    _fields_ = [("embedding", c_double * 1024)]

class ScheduleDecision(Structure):
    _pack_ = 1
    _fields_ = [("decision", c_uint32)]

class FeedBack(Structure):
    _pack_ = 1
    _fields_ = [("reward", c_double)]

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

  def connect(self, host, port):
    self.ssock.bind((host, port))
    self.ssock.listen(3)
    print("Server listening on port {:d}".format(port))

    self.csock, self.client_address = self.ssock.accept()
    print("Accepted connection from {:s}".format(self.client_address[0]))

  def fetch_req(self) -> Reqs:
    #if self.fired:
    #  buff = self.csock.recv(sizeof(FeedBack))
    #  feedback = FeedBack.from_buffer_copy(buff)
    #  reward = feedback.reward

    tensor_lists = []
    buff = self.csock.recv(sizeof(RequestEmbed))

    while buff:
      payload_in = RequestEmbed.from_buffer_copy(buff)
      embedding = np.array(payload_in.embedding)
      if np.isnan(embedding).all():
        break

      #Set invalid embedding to 0
      embedding[np.isnan(embedding)] = 0.0

      tensor_lists.append(embedding)
      buff = self.csock.recv(sizeof(RequestEmbed))


    print("Received {:d} tensors".format(len(tensor_lists)))
    if self.fired:
      reward = -(time.time() - self.prev_time) * len(tensor_lists)
      self.prev_time = time.time()
      return reward, Reqs(np.array(tensor_lists), np.array([0]), self.csock)

    self.fired = True
    self.prev_time = time.time()
    return Reqs(np.array(tensor_lists), np.array([0.0]), self.csock)

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
    engine.connect("localhost", 2300)
    reqs = engine.fetch_req()
    while(True):
      reqs.make_decision(reqs.get_random_action())
      reward, reqs = engine.fetch_req()
      print(reward)
  except:
    print("Error")
    engine.ssock.close()
    raise
