import os
import threading
import sys
import io
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from connector import *
from tensorboardX import GlobalSummaryWriter
from typing import NamedTuple
from transactions import *
import torch.multiprocessing as mp
from concurrent_test import BatchRunner
from exp_collect import Episode, Experience
from model import MyModel
from tqdm import tqdm
from sequence_generator import SequenceGenerator
import zmq
import json

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ivm"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Postgresql_ivm-v1"
    """the id of the environment"""
    total_episodes: int = 200
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 2000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    featurizer_path: str = "/workspaces/LearnedIVM/pg_ivm_learned/py/featurizer.pkl"
    """the path of the featurizer"""
    actor_learning_rate: float = 1e-3
    """the learning rate of the actor"""
    value_network_learning_rate: float = 1e-3
    """the learning rate of the value network"""
    model_save_frequncey: int = 20
    """save the model every `model_save_frequncey` episodes"""


class DecisionServer:
    def __init__(self, model: MyModel, start_next: mp.Barrier):
        self.process = mp.Process(
            target=self.decision_func, args=(model, start_next)
        )

    def start(self):
        self.process.start()

    def stop(self):
        self.process.kill()

    def decision_func(
        self, model: MyModel, start_next: mp.Barrier
    ):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")

        # Important! Set the number of threads to 1 to avoid potential performance issues
        torch.set_num_threads(1)
        episode = Episode()
        entropies = []
        episode_idx = 0

        while True:
            msg = str(socket.recv().decode("ascii"))
            request = json.loads(msg.replace("'", '"'))
            if request["type"] == "schedule":
                seq = SequenceGenerator.get()
                probs = model.predict(
                    request["running_plans"], request["candidate_plans"]
                ).flatten()
                category = torch.distributions.Categorical(probs)
                action = category.sample().item()
                entropies.append(category.entropy().item())
                #print(f"{seq} action {action} selected")
                episode.add(
                    {
                        "candidate_plans": request["candidate_plans"],
                        "running_plans": request["running_plans"],
                    },
                    action,
                    seq,
                )
                socket.send_string(json.dumps({"seq": seq, "action": action}))
            elif request["type"] == "reward":
                #print(f"{request['seq']} reward received")
                episode.set_rwd(request["seq"], request["reward"])
                socket.send_string("ack")
            elif request["type"] == "done":
                episode.finish_up()
                actor_loss, vf_loss = model.train(episode)
                print("Trainer: Episode Training Done")
                writer.add_scalar("episode/entropies", np.mean(entropies), episode_idx)
                writer.add_scalar(
                    "episode/total_reward", episode.get_reward_sum(), episode_idx
                )
                writer.add_scalar("episode/actor_loss", actor_loss, episode_idx)
                writer.add_scalar("episode/vf_loss", vf_loss, episode_idx)
                print(
                    f"Episode {episode_idx} States, Entropy: {np.mean(entropies)}, Total Reward: {episode.get_reward_sum()}, Actor Loss: {actor_loss}, VF Loss: {vf_loss}"
                )
                socket.send_string("ack")
                start_next.wait()
                if episode_idx % Args.model_save_frequncey == 0:
                    model.save(f"runs/{run_name}/model_{episode_idx}.ckpt")
                episode_idx += 1
                episode = Episode()
                entropies = []
            elif request["type"] == "drop":
                episode = Episode()
                entropies = []
                socket.send_string("ack")
                start_next.wait()


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = Args.torch_deterministic


if __name__ == "__main__":

    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = GlobalSummaryWriter(log_dir=f"runs/{run_name}", flush_secs=30)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    fix_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model = MyModel(
        args.gamma, args.actor_learning_rate, args.value_network_learning_rate
    )
    model.load_tree_featurizer(args.featurizer_path)
    model.share_memory()
    print("Model loaded!")

    start_next: mp.Barrier = mp.Barrier(2)

    decision_server = DecisionServer(model,start_next)
    decision_server.start()
    print("Decision server started!")

    context = zmq.Context()

    for i in range(args.total_episodes):
        wl = [[shipment]]
        runner = BatchRunner(wl * 8)
        runner.start()
        print("Runner started!")
        results = runner.get_result()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")
        if -1 in results:
            socket.send_string("{'type': 'drop'}")
            print("Episode Dropped")
        else:
            socket.send_string("{'type': 'done'}")
            writer.add_scalar("episode/average_time_cost", np.mean(results), i)
            print(f"Episode Done in {np.mean(results)}")
        socket.close()
        start_next.wait()

    writer.close()
