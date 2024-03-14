import zmq
import sequence_generator
import json
import pickle
from typing import NamedTuple

class Experience(NamedTuple):
    state: dict
    action: str
    reward: float


class Episode:
    def __init__(self):
        self.episode = []
        self.idx2id = {}
        self.id2idx = {}

    def add(self, state, action, seq):
        self.episode.append((state, action))
        self.idx2id[len(self.episode) - 1] = seq
        self.id2idx[seq] = len(self.episode) - 1

    def set_rwd(self, id, rwd):
        idx = self.id2idx[id]
        ori = self.episode[idx]
        self.episode[idx] = (*ori, rwd)

    def finish_up(self):
        for tup in self.episode:
            if len(tup) == 2:
                raise ValueError("Episode not finished")
        self.exp = [Experience(*tup) for tup in self.episode]
        del self.episode
        del self.idx2id
        del self.id2idx


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    epi_index = 0

    episode = Episode()

    while True:
        msg = str(socket.recv().decode("ascii"))
        request = json.loads(msg.replace("'", '"'))

        if request["type"] == "schedule":
            seq = sequence_generator.SequenceGenerator.get()
            episode.add(
                {
                    "candidate_plans": request["candidate_plans"],
                    "running_plans": request["running_plans"],
                },
                request["action"],
                seq
            )
            # send back the sequence
            socket.send_string(json.dumps({"seq": seq}))
            print(f"SEQ:{seq} Logged!")
        elif request["type"] == "reward":
            seq = request["seq"]
            rwd = request["reward"]
            episode.set_rwd(seq, rwd)
            socket.send_string("ack")
            print(f"SEQ:{seq} RWD:{rwd} Logged!")
        elif request["type"] == "done":
            episode.finish_up()
            socket.send_string("ack")
            with open(f"episode_{epi_index}.pkl", "wb") as f:
                pickle.dump(episode, f)
            epi_index += 1
            episode = Episode()

