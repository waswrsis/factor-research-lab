import numpy as np
import pandas as pd
import pytest

from factorlab.dataset import Dataset
from factorlab.experiment import starter_graph
from factorlab.graph import validate_graph
from factorlab.protocol import Protocol


@pytest.mark.parametrize('lookback', [1, 4, 12, 48, 168])
def test_supported_protocol_lookbacks_build_valid_graphs(lookback):
    dataset = Dataset(np.zeros((0, lookback, 2), np.float32), np.empty(0), np.empty(0),
                      pd.DataFrame(), ('return_1h', 'volume_z_24h'))
    graph = starter_graph(dataset, Protocol(lookback=lookback))
    validate_graph(graph)
    assert all(max(e.scales) <= lookback for e in graph.experts)
