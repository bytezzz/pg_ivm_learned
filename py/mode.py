import torch.nn as nn
import torch
import pickle
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
from TreeConvolution.util import prepare_trees
from exp_collect import Episode
from featurelize import TreeFeaturizer

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
        running = self.first_linear(running)
        running = nn.functional.relu(running)
        running = torch.sum(running, dim=0)

        executing = self.first_linear(executing)
        executing = nn.functional.relu(executing)
        executing = torch.sum(executing, dim=0)

        X = torch.cat((running, executing), 0)

        X = self.second_linear(X)
        X = nn.functional.relu(X)
        X = self.fc(X)
        return X

    def cuda(self):
        self.__cuda = True
        return super().cuda()


class MyModel:
    def __init__(self) -> None:
        self.tree_featurizer = TreeFeaturizer()
        self.query_plan_conv = QueryPlanTreeConv(self.tree_featurizer.num_operators() + 3)
        self.global_graph_conv = GlobalGraphConv(self.query_plan_conv.out_channels())
        self.evaluator = nn.Linear(self.query_plan_conv.out_channels() + self.global_graph_conv.out_channels(), 1)

    def fit_trees(self, trees):
        self.tree_featurizer.fit(trees)

    def predict(self, executing, candidate):
        executing_trees = self.query_plan_conv(
            self.tree_featurizer.transform(executing)
        )
        candidate_trees = self.query_plan_conv(
            self.tree_featurizer.transform(candidate)
        )

        global_embedding = self.global_graph_conv(executing_trees, candidate_trees)

        global_embedding = global_embedding.expand(len(candidate_trees), -1)
        embeddings = torch.cat((global_embedding, candidate_trees), 1)

        x = self.evaluator(embeddings)

        return nn.functional.softmax(x, dim=0)

