import networkx as nx
import numpy as np
import copy


class Edge(object):
    idCounter = 0

    def __init__(self, u, v, c=0):
        self.source = u
        self.sink = v
        self.capacity = c
        self.edgeid = Edge.idCounter
        Edge.idCounter += 1
        self.fvalue = []
        # self.init_load = 0

    def __repr__(self):
        return "Edge%s:%s->%s,%s" % (self.edgeid, self.source, self.sink, self.capacity)

    @classmethod
    def reset(self):
        Edge.idCounter = 0


class Flow(object):
    idCounter = 0

    def __init__(self, u, v, d):
        self.source = u
        self.sink = v
        self.demand = d
        self.flowid = Flow.idCounter
        Flow.idCounter += 1
        self.shorest_paths = []
        self.paths = None
        # self.paths = np.zeros(num_edges)

    def __repr__(self):
        return "%s->%s,%s" % (self.source, self.sink, np.round(self.demand, decimals=2))

    @classmethod
    def reset(self):
        Flow.idCounter = 0


class srEnv():
    def __init__(self, graph, capacity=None):
        self.graph = graph.to_directed()
        self.capacity = capacity
        self.num_edges = graph.number_of_edges()
        # self.init_load = None
        # self.init_utils = None
        self.num_flows = self.graph.number_of_nodes() * (self.graph.number_of_nodes()-1)
        self.steps = 0
        self.init_edges()
        self.init_flows()
        self.updatef()
        self.not_getting_better = 0

    def init_edges(self):
        Edge.reset()
        self.edgedict = {}
        self.edges = []
        for eid, e in enumerate(self.graph.edges):
            if self.capacity is None:
                capacity = 100
            else:
                capacity = self.capacity[eid]

            edge = Edge(e[0], e[1], capacity)
            edge.load = np.zeros(self.num_flows)
            self.edges.append(edge)
            self.edgedict[e] = edge
            edge.fvalue = [0] * self.num_flows

    def reset(self, traffic):
        self.tm = traffic
        self.nonzero_flows = []
        self.init_state = None
        # num_nodes = self.graph.number_of_nodes()
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        for flow in self.flows:
            flow.demand = self.tm[flow.source][flow.sink]
            if flow.demand > 0:
                self.nonzero_flows.append(flow)
        self._reset()
        return self.get_state()

    def init_flows(self):
        Flow.reset()
        self.flows = []
        self.fdict = {}
        lengths = list(nx.all_pairs_shortest_path_length(self.graph))
        min_length = 1
        pair_to_length_dict = {}
        for x, yy in lengths:
            for y, l in yy.items():
                if l >= min_length:
                    pair_to_length_dict[x, y] = l
        node_pairs = list(pair_to_length_dict)
        for n in node_pairs:
            # demand = self.tm[n[0]][n[1]]
            flow = Flow(n[0], n[1], 0)
            flow.paths = np.zeros(self.num_edges)
            self.flows.append(flow)
            self.fdict[n] = flow

    def updatef(self):
        # print("updating f")
        for flow in self.flows:
            paths = [p for p in nx.all_shortest_paths(
                self.graph, source=flow.source, target=flow.sink, weight="weight")]
            # flow.paths = paths
            for path in paths:
                edges = list(zip(path, path[1:]))
                for e in edges:
                    edge = self.edgedict[e]
                    edge.fvalue[flow.flowid] += 1/len(paths)
                    if edge not in flow.shorest_paths:
                        flow.shorest_paths.append(edge)
                        flow.paths[edge.edgeid] = 1/len(paths)

    def getf(self, source, target, edge):
        flow = self.fdict[(source, target)]
        return edge.fvalue[flow.flowid]

    def getg(self, source, target, intermediate, edge):
        return self.getf(source, intermediate, edge) + self.getf(intermediate, target, edge)

    # 初始状态 shortest path
    # return all link utilization
    def _reset(self):
        self.steps = 0
        self.not_getting_better = 0
        for edge in self.edges:
            edge.load = np.zeros(self.num_flows)
        if self.init_state is None:
            for edge in self.edges:
                for flow in self.flows:
                    if flow.demand == 0:
                        continue
                    fvalue = edge.fvalue[flow.flowid]
                    if fvalue > 0:
                        # flow.edges.append(edge)
                        edge.load[flow.flowid] += fvalue * flow.demand
            self.init_state = copy.deepcopy(self.edges)
        else:
            self.edges = copy.deepcopy(self.init_state)
        self.init_utils = self.get_utils()
        self.prev_utils = self.init_utils

    # 添加一个中间节点
    # new_state, reward, done, _ = env.step(action)
    # 现在假设为2-sr
    def updateloads(self, flow, segmentid):
        # flow, k = action
        segmentid = segmentid.item()
        flow1 = self.fdict[(flow.source, segmentid)]
        flow2 = self.fdict[(segmentid, flow.sink)]
        # load = self.prev_loads
        current_paths = np.where(flow.paths > 0)[0]
        current_paths = [self.edges[eid] for eid in current_paths]
        for edge in current_paths:
            edge.load[flow.flowid] = 0
            flow.paths[edge.edgeid] = 0
        for edge in flow1.shorest_paths:
            fvalue = edge.fvalue[flow1.flowid]
            edge.load[flow.flowid] += fvalue * flow.demand
            flow.paths[edge.edgeid] += fvalue
        for edge in flow2.shorest_paths:
            fvalue = edge.fvalue[flow2.flowid]
            edge.load[flow.flowid] += fvalue * flow.demand
            flow.paths[edge.edgeid] += fvalue

    def get_utils(self):
        init_load = np.array([sum(edge.load) for edge in self.edges])
        capacity = np.array([edge.capacity for edge in self.edges])
        return init_load/capacity

    def step(self, flowid, segmentid):
        # for flowid, action in enumerate(actions):
        flow = self.flows[flowid]
        # print(flow)
        # flow, k = action

        done = False
        current_load = [edge.load for edge in self.edges]
        current_routing = [flow.paths for flow in self.flows]
        # prev_utils = self.prev_utils
        if flow.source == segmentid or flow.sink == segmentid:
            self.not_getting_better += 1
            if self.not_getting_better > 20:
                done = True
            return self.prev_utils, 0, done
        self.updateloads(flow, segmentid)
        util = self.get_utils()
        reward = self.reward_func(util)

        # not getting better roll back
        if reward > 1:
            self.not_getting_better += 1
        else:
            self.not_getting_better += 1
            util = self.prev_utils
            for edgeid, edge in enumerate(self.edges):
                edge.load = current_load[edgeid]
            for flowid, flow in enumerate(self.flows):
                flow.paths = current_routing[flowid]

        self.prev_utils = util
        flow_routing = self.get_state()
        if self.not_getting_better > 100:
            done = True

        return (util, flow_routing), reward, done

    def reward_func(self, util):
        max_util_diff = max(self.prev_utils) - max(util)
        std_diff = np.std(self.prev_utils) - np.std(util)
        # print("maxu diff", max_util_diff, "stddiff", std_diff)
        reward = 1000 * max_util_diff + 1000 * std_diff
        # if max_util_diff > 0 or std_diff > 0:
        #     reward += 10
        reward = np.exp(reward)
        # print(reward)
        return reward

    def get_state(self):
        flowstate = []
        for flow in self.flows:
            # flowstate = np.append(flowstate, flow.demand)
            flowstate = np.append(flowstate, flow.paths * flow.demand)
        return flowstate


def main():
    seed = 1
    # num_flows = 50
    num_nodes = 20
    num_edges = 100

    graph = nx.gnm_random_graph(num_nodes, num_edges, seed).to_directed()
    capacity = np.array([100]*num_edges*2)
    env = srEnv(graph, capacity)
    env.updateloads(env.flows[56], 14)
    env.get_state()


if __name__ == "__main__":
    main()
