import os
import threading
import sys
import io
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from connector import *
from torch.utils.tensorboard import SummaryWriter
from typing import NamedTuple
import torch.multiprocessing as mp
from concurrent_test import run_batch
from alive_progress import alive_bar
import tqdm
import psutil


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ivm"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Postgresql_ivm-v1"
    """the id of the environment"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 800
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 50
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 800
    """timestep to start learning"""
    train_frequency: int = 200
    """the frequency of training"""
    evalutate_frequency: int = 800


ACTION_FEATURE_SIZE = 1024
ENV_FEATURE_SIZE = 17
MAX_ACTION_NUM = 100


class RequestsServer:
    def __init__(self):
        self.control_queue = mp.Queue()
        self.result = mp.Value("d", 0.0)
        self.bootstrap_process = mp.Process(target=run_batch, args=(self.result, self.control_queue, False))

    def start(self):
        self.bootstrap_process.start()

    def kill(self):
        if not self.bootstrap_process.is_alive():
            return

        self.control_queue.put("shutdown")

        with alive_bar(
            title="Waiting unfinished backend processes to exit",
            spinner="classic",
            bar=None,
            monitor=False,
            elapsed=True,
            stats=False,
        ):
            self.bootstrap_process.join()
            #while not os.system('ps -ef | grep "[p]ostgres: vscode" > /dev/null'):
            #    time.sleep(0.5)

        self.control_queue = mp.Queue()
        self.bootstrap_process = mp.Process(target=run_batch, args=(self.result, self.control_queue, False))


class DecisionServer:
    def __init__(self, engine, model, epsilon):
        self.process = mp.Process(target=self.decision_func, args=(engine, model))
        self.epsilon = epsilon

    def start(self):
        self.process.start()

    def stop(self):
        self.process.kill()

    def decision_func(self, engine, model):

        # Important! Set the number of threads to 1 to avoid potential performance issues
        torch.set_num_threads(1)

        while True:
            _, reqs = engine.fetch_req()
            if random.random() < epsilon:
                actions = reqs.get_random_action()
            else:
                env_obs = (
                    torch.from_numpy(reqs.get_env_features())
                    .to(torch.float32)
                    .unsqueeze(dim=0)  # Add env dimension
                )
                action_obs = (
                    torch.from_numpy(reqs.get_action_features())
                    .to(torch.float32)
                    .unsqueeze(dim=0)  # Add env dimension
                )
                with torch.no_grad():
                    q_values = model(
                        env_obs.unsqueeze(dim=0),  # Add batch dimension
                        action_obs.unsqueeze(dim=0),  # Add batch dimension
                    )
                actions = torch.argmax(q_values.squeeze(dim=0), dim=1).cpu().numpy()
            reqs.make_decision(actions.flatten()[0])


class DBReplayBufferSamples(NamedTuple):
    env_observations: torch.Tensor
    action_space_observations: torch.Tensor
    next_env_observations: torch.Tensor
    next_action_space_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class DBReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        n_envs: int,
        max_action_num: int,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_action_num = max_action_num
        self.device = device

        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

        self.env_observations = np.zeros(
            (buffer_size, n_envs, observation_dim), dtype=np.float32
        )
        self.action_space_observations = np.zeros(
            (buffer_size, n_envs, max_action_num, action_dim), dtype=np.float32
        )
        self.next_env_observations = np.zeros(
            (buffer_size, n_envs, observation_dim), dtype=np.float32
        )
        self.next_action_space_observations = np.zeros(
            (buffer_size, n_envs, max_action_num, action_dim), dtype=np.float32
        )
        self.actions = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.valid_action_num = np.zeros((buffer_size, n_envs), dtype=np.uint32)
        self.next_valid_action_num = np.zeros((buffer_size, n_envs), dtype=np.uint32)

    def add(
        self,
        env_observations: np.ndarray,
        action_space_observations: np.ndarray,
        next_env_observations: np.ndarray,
        next_action_space_observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        assert env_observations.shape == (self.n_envs, self.observation_dim)
        assert next_env_observations.shape == (self.n_envs, self.observation_dim)
        assert rewards.shape == (self.n_envs,)
        assert dones.shape == (self.n_envs,)

        self.env_observations[self.pos] = env_observations
        self.next_env_observations[self.pos] = next_env_observations
        self.action_space_observations[
            self.pos, :, : action_space_observations.shape[1], :
        ] = action_space_observations
        self.next_action_space_observations[
            self.pos, :, : next_action_space_observations.shape[1], :
        ] = next_action_space_observations
        self.valid_action_num[self.pos] = action_space_observations.shape[1]
        self.next_valid_action_num[self.pos] = next_action_space_observations.shape[1]

        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.buffer_size, size=batch_size)

        return DBReplayBufferSamples(
            torch.from_numpy(self.env_observations[idxs]).to(self.device),
            torch.from_numpy(self.action_space_observations[idxs, :, :, :]).to(
                self.device
            ),
            torch.from_numpy(self.next_env_observations[idxs]).to(self.device),
            torch.from_numpy(self.next_action_space_observations[idxs, :, :, :]).to(
                self.device
            ),
            torch.from_numpy(self.actions[idxs]).to(self.device),
            torch.from_numpy(self.rewards[idxs]).to(self.device),
            torch.from_numpy(self.dones[idxs]).to(self.device),
        )


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                ENV_FEATURE_SIZE + ACTION_FEATURE_SIZE,
                120,
            ),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

    def forward(
        self,
        env_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """
        :param env_embedding: (BATCH_SIZE, ENGINE_NUMS, ENV_FEATURE_SIZE)
        :param action_embedding: (BATCH_SIZE, ENGINE_NUMS, VALID_ACTIONS_NUMS, ACTION_FEATURE_SIZE)
        """

        # Assert the shape of the embeddings to ensure they match the expected sizes
        assert (
            env_embedding.shape[2] == ENV_FEATURE_SIZE
        ), "The env_embedding does not have the correct shape."
        assert (
            action_embedding.shape[3] == ACTION_FEATURE_SIZE
        ), "The action_embedding does not have the correct shape."

        batch_size = env_embedding.shape[0]
        engine_nums = env_embedding.shape[1]
        max_actions = action_embedding.shape[2]

        # Optionally normalize the embeddings if needed
        # env_embedding = torch.nn.functional.normalize(env_embedding, dim=2)
        # action_embedding = torch.nn.functional.normalize(action_embedding, dim=3)

        # Expand env_embedding to match the action_embedding's shape for concatenation
        expanded = env_embedding.unsqueeze(2).expand(-1, -1, max_actions, -1)

        # Concatenate the embeddings along the last dimension and reshape for processing
        compact_features = torch.cat((expanded, action_embedding), dim=3).reshape(
            batch_size * engine_nums * max_actions, -1
        )

        # Process compact_features through the network and reshape the output to match the expected format
        q_values = self.network(compact_features).reshape(
            (batch_size, engine_nums, max_actions)
        )

        return q_values


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = Args.torch_deterministic


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluate_model(evaluate_func):
    print("Trying to evaluate")
    time_cost = mp.Value("d", 0.0)
    control_queue = mp.Queue()
    evaluator = mp.Process(target=evaluate_func, args=(time_cost,control_queue, True))

    evaluator.start()
    evaluator.join()

    return time_cost.value


if __name__ == "__main__":

    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

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

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    fix_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    q_network = QNetwork().to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    q_network.share_memory()
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = DBReplayBuffer(
        args.buffer_size,
        ENV_FEATURE_SIZE,
        ACTION_FEATURE_SIZE,
        1,
        MAX_ACTION_NUM,
        device,
    )

    start_time = time.time()

    pg_engine = Engine()
    pg_engine.bind("localhost", 2300)

    # make sure postgres is running
    time.sleep(5)

    requests_server = RequestsServer()
    requests_server.start()

    reqs = pg_engine.fetch_req()

    bar = tqdm.tqdm(
        total=args.learning_starts, ncols=0, desc="Collecting initial experience"
    )

    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps):

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        env_obs = (
            torch.from_numpy(reqs.get_env_features())
            .to(torch.float32)
            .to(device)
            .unsqueeze(dim=0)  # Add env dimension
        )
        action_obs = (
            torch.from_numpy(reqs.get_action_features())
            .to(torch.float32)
            .to(device)
            .unsqueeze(dim=0)  # Add env dimension
        )

        if random.random() < epsilon:
            actions = reqs.get_random_action()
        else:
            q_values = q_network(
                env_obs.unsqueeze(dim=0),  # Add batch dimension
                action_obs.unsqueeze(dim=0),  # Add batch dimension
            )
            actions = torch.argmax(q_values.squeeze(dim=0), dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        reqs.make_decision(actions.flatten()[0])

        reward, next_reqs = pg_engine.fetch_req()

        # # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = np.zeros_like(next_obs[idx]).fill(-np.inf)

        rb.add(
            np.array([reqs.get_env_features()], dtype=np.float32),
            np.array([reqs.get_action_features()], dtype=np.float32),
            np.array([next_reqs.get_env_features()], dtype=np.float32),
            np.array([next_reqs.get_action_features()], dtype=np.float32),
            np.array([actions]),
            np.array([reward]),
            np.array([0.0]),
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        reqs = next_reqs
        bar.update(1)

        if global_step < args.learning_starts:
            continue

        if global_step == args.learning_starts:
            bar.close()
            bar = tqdm.tqdm(total=args.evalutate_frequency, ncols=0, desc="Training")

        # ALGO LOGIC: training.
        if global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                target_max, _ = target_network(
                    data.next_env_observations,
                    data.next_action_space_observations
                ).max(dim=2)
                td_target = (data.rewards + args.gamma * target_max).squeeze()

            old_val = (
                q_network(data.env_observations, data.action_space_observations)
                .gather(2, data.actions.unsqueeze(dim=2))
                .squeeze()
            )
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update target network
        if global_step % args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(
                target_network.parameters(), q_network.parameters()
            ):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data
                    + (1.0 - args.tau) * target_network_param.data
                )

        if global_step % args.evalutate_frequency == 0:

            # Since previous requests have benn fetched, we give a random action to make it continue
            reqs.make_decision(reqs.get_random_action().flatten()[0])

            # Start a decision server before stopping the benchmark server
            # Or the requests will be blocked
            decision_server = DecisionServer(pg_engine, q_network, epsilon)
            decision_server.start()
            requests_server.kill()

            pg_engine.flush()

            print("Evaluating Model")
            time_cost = evaluate_model(run_batch)
            decision_server.stop()

            print(f"Evaluate finish! Time Cost: {time_cost}")
            writer.add_scalar("time_cost", time_cost, global_step)

            # Restart the request server to continue the training
            requests_server.start()
            bar.reset()

            # Drop previous reward due to the incorrect time interval
            _, reqs = pg_engine.fetch_req()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    writer.close()
