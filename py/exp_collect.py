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
        self.done = False
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

    def __getitem__(self, idx):
        if not self.done:
            raise ValueError("Episode not finished yet!")
        return self.exp[idx]

    def __len__(self):
        if not self.done:
            raise ValueError("Episode not finished yet!")
        return len(self.exp)

    def get_reward_sum(self):
        return self.reward_sum


    def finish_up(self):
        self.exp = []
        reward_sum = 0
        for tup in self.episode:
            if len(tup) == 3:
                reward_sum += tup[2]
                self.exp.append(Experience(*tup))
        del self.episode
        del self.idx2id
        del self.id2idx
        self.done = True
        self.reward_sum = reward_sum


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
            print(f'Episode{epi_index} done!')
            episode.finish_up()
            socket.send_string("ack")
            with open(f"exps/episode_{epi_index}.pkl", "wb") as f:
                pickle.dump(episode, f)
            epi_index += 1
            episode = Episode()
        elif request["type"] == "drop":
            print("Dropping episode, starting a new one.")
            episode = Episode()
            socket.send_string("ack")

