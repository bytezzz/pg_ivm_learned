import torch.nn as nn
import torch
import pickle
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
from TreeConvolution.util import prepare_trees
from exp_collect import Episode
from featurelize import TreeFeaturizer
from itertools import chain


def left_child(x):
    if len(x) != 3:
        return None
    return x[1]


def right_child(x):
    if len(x) != 3:
        return None
    return x[2]


def features(x):
    return x[0]


class QueryPlanTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(QueryPlanTreeConv, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False
        self.__out_channels = out_channels

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, out_channels),
        )

    def in_channels(self):
        return self.__in_channels

    def forward(self, x):
        trees = prepare_trees(x, features, left_child, right_child, cuda=self.__cuda)
        return self.tree_conv(trees)

    def cuda(self):
        self.__cuda = True
        return super().cuda()

    def out_channels(self):
        return self.__out_channels


class GlobalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(GlobalGraphConv, self).__init__()
        self.__cuda = False
        self.__out_channels = out_channels

        self.first_linear = nn.Linear(in_channels, 128)
        self.second_linear = nn.Linear(256, 128)
        self.fc = nn.Linear(128, out_channels)

    def in_channels(self):
        return self.__in_channels

    def out_channels(self):
        return self.__out_channels

    def forward(self, running, executing):
        # X: (num_nodes, in_channels)
        if running is not None:
            running = self.first_linear(running)
            running = nn.functional.relu(running)
            running = torch.sum(running, dim=0)
        else:
            running = torch.zeros(128)

        if executing is not None:
            executing = self.first_linear(executing)
            executing = nn.functional.relu(executing)
            executing = torch.sum(executing, dim=0)
        else:
            executing = torch.zeros(128)

        X = torch.cat((running, executing), 0)

        X = self.second_linear(X)
        X = nn.functional.relu(X)
        X = self.fc(X)
        return X

    def cuda(self):
        self.__cuda = True
        return super().cuda()


class ValueNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(ValueNetwork, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False
        self.__out_channels = out_channels

        self.net = nn.Sequential(
            nn.Linear(self.__in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.__out_channels),
        )

    def in_channels(self):
        return self.__in_channels

    def out_channels(self):
        return self.__out_channels

    def forward(self, x):
        return self.net(x)

    def cuda(self):
        self.__cuda = True
        return super().cuda()


class MyModel:
    def __init__(self, discount, actor_lr, vf_lr) -> None:
        self.tree_featurizer = TreeFeaturizer()
        self.query_plan_conv = QueryPlanTreeConv(
            self.tree_featurizer.num_operators() + 3
        )
        self.global_graph_conv = GlobalGraphConv(self.query_plan_conv.out_channels())
        self.evaluator = nn.Linear(
            self.query_plan_conv.out_channels() + self.global_graph_conv.out_channels(),
            1,
        )
        self.value_network = ValueNetwork(self.global_graph_conv.out_channels(), 1)
        self.discount_factor = discount
        self.actor_optimizer = torch.optim.Adam(
            chain(
                self.query_plan_conv.parameters(),
                self.global_graph_conv.parameters(),
                self.evaluator.parameters(),
            ),
            lr=actor_lr,
        )
        self.vf_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=vf_lr)

    def load_tree_featurizer(self, path):
        with open(path, "rb") as f:
            self.tree_featurizer = pickle.load(f)

    def fit_trees(self, trees):
        self.tree_featurizer.fit(trees)

    def share_memory(self):
        self.query_plan_conv.share_memory()
        self.global_graph_conv.share_memory()
        self.evaluator.share_memory()
        self.value_network.share_memory()

    def embed_state(self, executing, candidate):
        if executing:
            executing_trees = self.query_plan_conv(
                self.tree_featurizer.transform(executing)
            )
        else:
            executing_trees = None

        if candidate:
            candidate_trees = self.query_plan_conv(
                self.tree_featurizer.transform(candidate)
            )
        else:
            candidate_trees = None

        global_embedding = self.global_graph_conv(executing_trees, candidate_trees)

        global_embedding = global_embedding.expand(len(candidate_trees), -1)

        return torch.cat((global_embedding, candidate_trees), 1)

    def predict(self, executing, candidate):
        embeddings = self.embed_state(executing, candidate)

        x = self.evaluator(embeddings)

        return nn.functional.softmax(x, dim=0)

    def save(self, path):
        torch.save(
            {
                "query_plan_conv": self.query_plan_conv.state_dict(),
                "global_graph_conv": self.global_graph_conv.state_dict(),
                "evaluator": self.evaluator.state_dict(),
                "value_network": self.value_network.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.query_plan_conv.load_state_dict(checkpoint["query_plan_conv"])
        self.global_graph_conv.load_state_dict(checkpoint["global_graph_conv"])
        self.evaluator.load_state_dict(checkpoint["evaluator"])
        self.value_network.load_state_dict(checkpoint["value_network"])

    def train(self, episode: Episode):
        trajectory_len = len(episode.exp)
        return_array = torch.zeros((trajectory_len,))
        g_return = 0.0

        action_t = torch.LongTensor([x.action for x in episode.exp])
        return_t = torch.FloatTensor([x.reward for x in episode.exp])

        for i in range(trajectory_len):
            g_return = episode.exp[i].reward + g_return * self.discount_factor
            return_array[i] = g_return

        state_t = []
        for step in episode.exp:
            embedding = self.embed_state(
                step.state["running_plans"], step.state["candidate_plans"]
            )
            global_embedding = embedding[0, : self.global_graph_conv.out_channels()]
            state_t.append(global_embedding)
        state_t = torch.stack(state_t)

        value_t = self.value_network(state_t.detach()).squeeze()
        with torch.no_grad():
            advantage_t = return_t - value_t

        padded_logits = torch.nn.utils.rnn.pad_sequence(
            [
                torch.FloatTensor(
                    self.predict(
                        step.state["running_plans"], step.state["candidate_plans"]
                    ).reshape(-1)
                )
                for step in episode.exp
            ],
            True
        )

        selected_action_prob = padded_logits.gather(1, action_t.unsqueeze(1)).squeeze()
        actor_loss = torch.mean(-torch.log(selected_action_prob) * advantage_t)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        vf_loss_fn = torch.nn.MSELoss()
        vf_loss = vf_loss_fn(value_t, return_t)
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.detach().cpu(), vf_loss.detach().cpu()
