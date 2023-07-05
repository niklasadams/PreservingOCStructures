import time
# import globals
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot as plt
from dgl.nn import GraphConv
import tensorflow as tf
import networkx as nx
import pandas as pd
import numpy as np
import dgl
import itertools
import os
import json

import ocpa.algo.predictive_monitoring.obj

# dd = os.path.join(dd,'config.json')
# with open(dd, "r") as config_file:
#     config_dict = json.load(config_file)
#     backend_name = config_dict.get('backend', '').lower()
# print(backend_name)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['DGLBACKEND'] = 'tensorflow'

# create graph dataset & labels for remaining time regression


def generate_sequential_graph_dataset(feature_graph_list, indices, ocel_object, k, target, include_last=True):
    idx = indices
    graph_list = []
    label_list = []
    for gid in tqdm(idx):

        # copy graph to prevent changes to feature storage
        graph = deepcopy(feature_graph_list[gid])

        # sort nodes according to timestamp
        instance_df = ocel_object.log.log.loc[[
            n.event_id for n in graph.nodes]].copy()
        instance_df = instance_df.sort_values('event_timestamp')

        node_id_map = {id: i for i, id in enumerate(instance_df.index)}

        # ENFORE SEQUENTIAL STRUCTURE
        index_series = instance_df["event_id"].to_list()
        edge_pairs = [(index_series[i], index_series[i+1])
                      for i in range(0, len(index_series)-1)]
        cart_set = [graph.nodes, graph.nodes]
        new_edges = [(n, m) for n, m in itertools.product(
            *cart_set) if (n.event_id, m.event_id) in edge_pairs]
        graph.replace_edges(new_edges)


        # define DGL graph
        dgl_graph = dgl.graph(
            data=([node_id_map[e.source] for e in graph.edges],
                  [node_id_map[e.target] for e in graph.edges]),
            num_nodes=len(graph.nodes)
        )

        # find correct node order in the graphs from feature storage
        sorted_node_indices = np.argsort(
            [node_id_map[n.event_id] for n in graph.nodes])

        # add features to each node from DGL graph
        event_indices = []
        labels = []
        features = []
        for idx in sorted_node_indices:
            node = graph.nodes[idx]
            node_label = node.attributes.pop(target)
            node_features = [v for _, v in node.attributes.items()]
            event_indices.append(node.event_id)
            labels.append(node_label)
            features.append(node_features)
        dgl_graph.ndata['event_indices'] = tf.constant(event_indices)
        dgl_graph.ndata['k'] = tf.constant(
            [k for i in range(0, len(event_indices))], dtype=tf.int64)
        dgl_graph.ndata['features'] = tf.constant(features, dtype=tf.float32)
        dgl_graph.ndata[target[0]] = tf.constant(labels, dtype=tf.float32)
        # extract subgraph and label for each node set as terminal node
        #k = 4
        if len(sorted_node_indices) != 0:
            correct = 0
            if not include_last:
                correct = 1
            for i in range(k - 1, len(sorted_node_indices)-correct):
                subgraph = dgl.node_subgraph(dgl_graph, nodes=range(
                    i - (k - 1), i + 1))  # include last event
                subgraph_label = subgraph.ndata[target[0]].numpy()[-1]
                graph_list.append(subgraph)
                label_list.append(subgraph_label)

    return graph_list, label_list


# create graph dataset & labels for remaining time regression
def generate_graph_dataset(feature_graph_list, indices, ocel_object, k, target, include_last=True):

    idx = indices
    graph_list = []
    label_list = []
    ext_time = 0
    sub_time = 0
    t_time = time.time()
    for gid in tqdm(idx):
        s_time = time.time()
        # copy graph to prevent changes to feature storage
        graph = deepcopy(feature_graph_list[gid])

        # sort nodes according to timestamp
        instance_df = ocel_object.log.log.loc[[
            n.event_id for n in graph.nodes]].copy()
        instance_df = instance_df.sort_values('event_timestamp')
        node_id_map = {id: i for i, id in enumerate(instance_df.index)}
        ext_time += time.time()-s_time
        # define DGL graph
        dgl_graph = dgl.graph(
            data=([node_id_map[e.source] for e in graph.edges],
                  [node_id_map[e.target] for e in graph.edges]),
            num_nodes=len(graph.nodes)
        )

        # find correct node order in the graphs from feature storage
        sorted_node_indices = np.argsort(
            [node_id_map[n.event_id] for n in graph.nodes])

        # add features to each node from DGL graph
        event_indices = []
        labels = []
        features = []
        for idx in sorted_node_indices:
            node = graph.nodes[idx]
            node_label = node.attributes.pop(target)
            node_features = [v for _, v in node.attributes.items()]
            event_indices.append(node.event_id)
            labels.append(node_label)
            features.append(node_features)
        dgl_graph.ndata['event_indices'] = tf.constant(event_indices)
        dgl_graph.ndata['k'] = tf.constant(
            [k for i in range(0, len(event_indices))], dtype=tf.int64)
        dgl_graph.ndata['features'] = tf.constant(features, dtype=tf.float32)
        dgl_graph.ndata[target[0]] = tf.constant(labels, dtype=tf.float32)
        # extract subgraph and label for each node set as terminal node
        correct = 0
        if not include_last:
            correct = 1
        for i in range(k - 1, len(sorted_node_indices) - correct):
            s_time = time.time()
            subgraph = dgl.node_subgraph(dgl_graph, nodes=range(
                i-(k-1), i+1))  # include last event

            subgraph_label = subgraph.ndata[target[0]].numpy()[-1]

            sub_time += time.time() - s_time
            # print(subgraph.ndata['event_indices'])
            graph_list.append(subgraph)
            label_list.append(subgraph_label)

    print(ext_time)
    print(sub_time)
    print(time.time()-t_time)
    return graph_list, label_list


# show table of events and features in graph
def get_ordered_event_list(graph):
    event_df = pd.DataFrame({
        'event_id': graph.ndata['event_indices'].numpy(),
        'label_remaining_time': graph.ndata['remaining_time'].numpy()
    })
    feature_df = pd.DataFrame(graph.ndata['features'].numpy())
    return {'events': event_df, 'features': feature_df}

# show graph with different labels on nodes


def visualize_graph(graph, add_str, labels='node_id', font_size=8, save=True):
    if save:
        f = plt.figure()
    nx_G = graph.to_networkx(node_attrs=['remaining_time', 'event_indices'])
    pos = nx.kamada_kawai_layout(nx_G)
    if labels == 'node_id':
        nx.draw(nx_G, pos, with_labels=True, node_color=[
                [.7, .7, .7]], font_size=font_size)
    elif labels == 'remaining_time':
        viz_labels = nx.get_node_attributes(nx_G, 'remaining_time')
        viz_labels = {k: np.round(v.numpy(), 9) for k, v in viz_labels.items()}
        nx.draw(nx_G, pos, labels=viz_labels, node_color=[
                [.7, .7, .7]], font_size=font_size)
    elif labels == 'event_indices':
        viz_labels = nx.get_node_attributes(nx_G, 'event_indices')
        viz_labels = {k: v.numpy() for k, v in viz_labels.items()}
        nx.draw(nx_G, pos, labels=viz_labels, node_color=[
                [.7, .7, .7]], font_size=font_size)
    if save:
        f.savefig("graph"+labels+add_str+".png")

# show ordered remaining times per event


def show_remaining_times(graph, plot=True):
    res = graph.ndata['remaining_time'].numpy()
    if plot:
        res = plt.plot(res)
    return res

# visualize graph instance


def visualize_instance(graph, txt, label):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1 = visualize_graph(graph, "", labels='node_id', save=False)
    ax2 = fig.add_subplot(222)
    ax2 = visualize_graph(graph, "", labels='event_indices', save=False)
    ax3 = fig.add_subplot(223)
    ax3 = visualize_graph(graph, "", labels='remaining_time', save=False)
    ax4 = fig.add_subplot(224)
    ax4 = show_remaining_times(graph, plot=True)
    #fig.suptitle('Label: Remaining time %s'%np.round(label.numpy(), 9))
    fig.savefig("instance"+txt+".png")

# custom data loader for yielding batches of graphs


class GraphDataLoader(tf.keras.utils.Sequence):

    def __init__(
        self,
        graph_list,
        graph_labels,
        batch_size,
        shuffle=True,
        add_self_loop=False,
        make_bidirected=False,
        on_gpu=False
    ):
        self.graph_list = graph_list
        self.graph_labels = graph_labels
        self.batch_size = batch_size
        self.add_self_loop = add_self_loop
        self.make_bidirected = make_bidirected
        self.indices = np.arange(0, len(graph_list))
        self.on_gpu = on_gpu
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.graph_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx *
                                     self.batch_size:(idx + 1) * self.batch_size]
        graph_batch = [self.graph_list[i] for i in batch_indices]
        if self.add_self_loop:
            graph_batch = [dgl.add_self_loop(g) for g in graph_batch]
        if self.make_bidirected:
            with tf.device('CPU:0'):
                graph_batch = [dgl.to_bidirected(
                    g, copy_ndata=True) for g in graph_batch]
        if self.on_gpu:
            graph_batch = [g.to('GPU:0') for g in graph_batch]
        dgl_batch = dgl.batch(graph_batch)

        if not np.all(dgl_batch.in_degrees() > 0):
            print('WARNING: 0-in-degree nodes found!')

        labels_batch = [self.graph_labels[i] for i in batch_indices]
        labels_batch = tf.stack(labels_batch, axis=0)

        return dgl_batch, labels_batch

# custom Graph Convolutional Network as tf.keras subclass for graph regression


class GCN(tf.keras.Model):

    def __init__(self, n_input_feats, n_hidden_feats):

        super().__init__()
        self.gconv_1 = GraphConv(n_input_feats, n_hidden_feats)
        self.gconv_2 = GraphConv(n_hidden_feats, n_hidden_feats)
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, g, input_features):
        h = self.gconv_1(g, g.ndata['features'])
        h = tf.keras.activations.gelu(h)
        h = self.gconv_2(g, h)
        h = tf.keras.activations.gelu(h)
        x = tf.reshape(
            h, (int(h.shape[0] / g.ndata['k'][0]), (g.ndata['k'][0] * h.shape[1])))
        out = self.dense(x)
        return out

# function to evaluate model on specific data loader


def evaluate_gnn(data_loader, gnn_model):
    predictions = []
    labels = []
    for batch_id in tqdm(range(data_loader.__len__())):
        dgl_batch, label_batch = data_loader.__getitem__(batch_id)
        pred = gnn_model(dgl_batch, dgl_batch.ndata['features']).numpy()
        predictions.append(pred)
        labels.append(label_batch.numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return predictions, labels

# custom Graph Convolutional Network as tf.keras subclass for graph classification


class ClassificationGCN(tf.keras.Model):

    def __init__(self, n_input_feats, n_hidden_feats, n_classes):

        super().__init__()
        self.gconv_1 = GraphConv(n_input_feats, n_hidden_feats)
        self.gconv_2 = GraphConv(n_hidden_feats, n_hidden_feats)
        self.dense = tf.keras.layers.Dense(n_classes, activation='linear')

    def call(self, g, input_features):
        h = self.gconv_1(g, g.ndata['features'])
        h = tf.keras.activations.gelu(h)
        h = self.gconv_2(g, h)
        h = tf.keras.activations.gelu(h)
        x = tf.reshape(
            h, (int(h.shape[0] / g.ndata['k'][0]), (g.ndata['k'][0] * h.shape[1])))
        out = self.dense(x)
        return out

# function to evaluate model on specific data loader


def evaluate_c_gnn(data_loader, gnn_model):
    predictions = []
    labels = []
    for batch_id in tqdm(range(data_loader.__len__())):
        dgl_batch, label_batch = data_loader.__getitem__(batch_id)
        pred = gnn_model(dgl_batch, dgl_batch.ndata['features']).numpy()
        predictions.append(pred)
        labels.append(label_batch.numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return predictions, labels
