"""
Implements Swendsen-Algorithm for delta Potts model
See https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.58.86
"""

from collections import namedtuple
from typing import List
import numpy as np
from rustworkx import PyGraph, connected_components
from libs.models.potts_model import PottsModel


SwendsenWangResult = namedtuple("SwendsenWangResult", [
    "sampled_states", "energy_log"
])


def construct_graph(model: PottsModel, temperature: float) -> PyGraph:
    """
    Assumes temperature is k_B T
    """
    neighbors = np.asarray(model.linked_info)
    delta_for_each_link = model.S[neighbors[:, 0]] == model.S[neighbors[:, 1]]
    edge_prob = np.where(delta_for_each_link, 1 - np.exp(-1/temperature), 0.0)
    is_edge_present = np.random.rand(*edge_prob.shape) < edge_prob

    graph = PyGraph(multigraph=False)
    graph.add_nodes_from(model.I)
    graph.add_edges_from([(link[0], link[1], 1) for link in neighbors[is_edge_present]])
    return graph


def resample_(model: PottsModel, ccs: List[set[int]]) -> np.ndarray:
    for cc in ccs:
        model.S[list(cc)] = np.random.randint(0, model.q)


def run_swendsen_wang(model: PottsModel, temperature: float, iters: int, save_period: int, energy_log_period: int) -> SwendsenWangResult:
    ret, energy_log = [], []
    for i in range(1, iters + 1):
        graph = construct_graph(model, temperature)
        ccs = connected_components(graph)
        resample_(model, ccs)
        if i % save_period == 0:
            ret.append(model.S)
            print(i)
        if energy_log_period > 0 and i % energy_log_period == 0:
            energy_log.append(model.Hamiltonian_2d_lattice_pbc())
    return SwendsenWangResult(ret, energy_log)
