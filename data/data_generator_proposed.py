"""
   Copyright 2023 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Only run as main
if __name__ != "__main__":
    raise RuntimeError("This script should not be imported!")

# Parse imports
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from typing import Tuple, Generator, Dict, Any, List
import numpy as np
import tensorflow as tf
from itertools import permutations
from re import sub
import argparse
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--test",
    action="store_true",
    help="If true, assume that the dataset is a test dataset. If so, there will not be "
    + "any training/validation partitioning or shuffling, and the dataset will not "
    + "check for unvalid delays.",
)
args = parser.parse_args()

# import datasets's DataNet API
sys.path.insert(0, args.input_dir)
from datanetAPI import DatanetAPI, TimeDist, Sample


def _get_network_decomposition(sample: Sample, sample_idx: int):
    """Given a sample from the DataNet API, it returns it as a sample for the model.

    Parameters
    ----------
    sample: Sample
        Sample from the DataNet API
    sample_idx : int
        Index of the sample

    Returns
    -------
    Tuple(dict, list)
        Tuple with the inputs of the model and the target variable to predict
    """
    # Read values from the DataNet API
    network_topology = sample.get_physical_topology_object()
    traffic_matrix = sample.get_traffic_matrix()
    physical_path_matrix = sample.get_physical_path_matrix()
    performance_matrix = sample.get_performance_matrix()
    packet_info_matrix = sample.get_pkts_info_object()
    _, sample_file_id = sample.get_sample_id()
    # Obtain links and nodes
    # Links are categorized in two types: links originating from routers or from switches
    # We also discard all links that start from the traffic generator
    router_links = dict()
    switch_links = dict()
    for edge in network_topology.edges:  # src, dst, port
        # We identify all traffic generators as the same port
        edge_id = sub(r"t(\d+)", "tg", network_topology.edges[edge]["port"])
        if edge_id.startswith("r"):
            selected_dict = router_links
        elif edge_id.startswith("s"):
            selected_dict = switch_links
        elif edge_id.startswith("tg"):
            continue
        else:
            raise ValueError(f"Unknown edge type: {edge_id}")
        selected_dict[edge_id] = {
            "capacity": float(network_topology.edges[edge].get("bandwidth", 1e9))
            / 1e9,  # original value is in bps, we change it to Gbps
            "node_id": int(edge_id[1]),
        }

    # In this scenario assume that flows can either follow CBR or MB distributions
    cbr_flows = dict()
    mb_flows = dict()
    used_links = set()  # Used later so we only consider used links
    # Add flows
    for src, dst in filter(
        lambda x: traffic_matrix[x]["AggInfo"]["AvgBw"] != 0
        and traffic_matrix[x]["AggInfo"]["PktsGen"] != 0,
        permutations(range(len(traffic_matrix)), 2),
    ):
        for local_flow_id in range(len(traffic_matrix[src, dst]["Flows"])):
            flow = traffic_matrix[src, dst]["Flows"][local_flow_id]
            # Size distribution is always determinstic
            # We must obtain the Interarrival packet gap (ipg)
            flow_packet_info = packet_info_matrix[src, dst][0][local_flow_id]
            # NOTE: don't use for x, _ when decomposing flow_packet_info, as the tuples
            # can have different lengths. Timestamps come in fractions of ns
            packet_timestamps = np.array([float(x[0]) for x in flow_packet_info])
            # Discard sample if no packet information is available
            if packet_timestamps.size < 2:
                return dict(), np.array([0.0])
            # Obtain packet level aggregation (per ms)
            # TODO: change aggregation level to an adjustable parameter
            packets_per_ms = [
                [0.0]
                for _ in np.arange(
                    0, packet_timestamps[-1] - packet_timestamps[0], 1000000
                )
            ] + [[0.0]]
            for tt in packet_timestamps:
                tt -= packet_timestamps[0]
                packets_per_ms[int(tt // 1000000)][0] += 1
            # Trim the sequence to be of a limited, fixed size
            # TODO: change sequence size to an adjustable parameter
            packets_per_ms = packets_per_ms[:1000]
            # Remove initial link from the path
            # We must also clean up the name of the traffic generator
            clean_og_path = [
                sub(r"t(\d+)", "tg", link)
                for link in physical_path_matrix[src, dst][2::2]
            ]

            # Prepare general flow info
            flow_id = f"{src}_{dst}_{local_flow_id}"
            flow_info = {
                "source": src,
                "destination": dst,
                "flow_id": flow_id,
                "length": len(clean_og_path),
                "og_path": clean_og_path,
                "traffic": traffic_matrix[src, dst]["AggInfo"]["AvgBw"],  # in bps
                "packets": traffic_matrix[src, dst]["AggInfo"]["PktsGen"],
                "packet_size": flow["SizeDistParams"]["AvgPktSize"],
                "packets_per_ms": packets_per_ms,
                "delay": performance_matrix[src, dst]["Flows"][local_flow_id][
                    "AvgDelay"
                ]
                * 1000,  # in ms
            }

            # Identify flow distribution, extract relevant characteristics
            if flow["TimeDist"] == TimeDist.CBR_T:
                flow_info["rate"] = flow["TimeDistParams"]["Rate"]

                cbr_flows[flow_id] = flow_info

            elif flow["TimeDist"] == TimeDist.MULTIBURST_T:
                flow_info["on_rate"] = flow["TimeDistParams"]["On_Rate"]
                flow_info["pkts_per_burst"] = flow["TimeDistParams"]["Pkts_per_burst"]
                flow_info["ibg"] = flow["TimeDistParams"]["IBG"]

                mb_flows[flow_id] = flow_info

            else:
                raise ValueError(f"Unknown flow distribution: {flow['TimeDist']}")

            # Add edges to the used_links set
            used_links.update(set(clean_og_path))

    # Purge unused links
    router_links = {kk: vv for kk, vv in router_links.items() if kk in used_links}
    switch_links = {kk: vv for kk, vv in switch_links.items() if kk in used_links}
    # After purging links, obtain node ids
    router_ids = set(vv["node_id"] for vv in router_links.values())
    switch_ids = set(vv["node_id"] for vv in switch_links.values())

    # Normalize flow naming
    # We give the indices in such a way that flows states are concatanated as [CBR, MB]
    ordered_cbr_flows = list()
    flow_mapping = dict()
    for idx, (flow_id, flow_params) in enumerate(cbr_flows.items()):
        flow_mapping[flow_id] = idx
        ordered_cbr_flows.append(flow_params)
    n_f = len(ordered_cbr_flows)
    ordered_mb_flows = list()
    for idx, (flow_id, flow_params) in enumerate(mb_flows.items()):
        flow_mapping[flow_id] = idx + n_f
        ordered_mb_flows.append(flow_params)
    ordered_flows = ordered_cbr_flows + ordered_mb_flows
    n_f = len(ordered_flows)

    # Normalize link naming
    ordered_router_links = list()
    router_link_mapping = dict()
    for idx, (link_id, link_params) in enumerate(router_links.items()):
        router_link_mapping[link_id] = idx
        ordered_router_links.append(link_params)
    n_l_r = len(ordered_router_links)
    ordered_switch_links = list()
    switch_link_mapping = dict()
    for idx, (link_id, link_params) in enumerate(switch_links.items()):
        switch_link_mapping[link_id] = idx
        ordered_switch_links.append(link_params)
    n_l_s = len(ordered_switch_links)
    # Normalize node naming
    # ordered_router_ids = list()
    router_id_mapping = dict()
    for idx, node_id in enumerate(router_ids):
        router_id_mapping[node_id] = idx
        # ordered_router_ids.append(node_id)
    n_r = len(router_ids)
    # ordered_switch_ids = list()
    switch_id_mapping = dict()
    for idx, node_id in enumerate(switch_ids):
        switch_id_mapping[node_id] = idx
        # ordered_switch_ids.append(node_id)
    n_s = len(switch_ids)

    # Obtain list of indices representing the topology
    # link_to_path: two dimensional array, first dimension are the paths, second dimension are the link indices
    # NOTE: when in the model, we assume that the links states are concatenated as [routers, switches]
    # meaning that the indices from the switch links must be increased by the number of router links (n_l_r)
    link_to_path = list()
    # We define link_pos_in_flows that will later help us build path_to_link
    link_pos_in_flows = list()
    for og_path in map(lambda x: x["og_path"], ordered_flows):
        # This list will contain the link indices in the original path,in order
        local_list = list()
        # This dict indicates for each link which are the positions in the original path, if any
        local_dict = dict()
        for link_id in og_path:
            # Transform link_id into a link index
            if link_id.startswith("r"):
                link_idx = router_link_mapping[link_id]
            elif link_id.startswith("s"):
                link_idx = switch_link_mapping[link_id] + n_l_r

            local_dict.setdefault(link_idx, list()).append(len(local_list))
            local_list.append(link_idx)
        link_to_path.append(local_list)
        link_pos_in_flows.append(local_dict)

    # path_to_r_link: two dimensional array, first dimension are the router links, second dimension are tuples.
    # Each tuple contains the path index and the link's position in the path
    # Note that a link can appear in multiple paths and multiple times in the same path
    path_to_r_link = list()
    for link_idx in range(n_l_r):
        local_list = list()
        for flow_idx in range(n_f):
            if link_idx in link_pos_in_flows[flow_idx]:
                local_list += [
                    (flow_idx, pos) for pos in link_pos_in_flows[flow_idx][link_idx]
                ]
        path_to_r_link.append(local_list)
    # path_to_s_link: two dimensional array, first dimension are the switch links, second dimension are tuples.
    # Each tuple contains the path index and the link's position in the path
    # Note that a link can appear in multiple paths and multiple times in the same path
    path_to_s_link = list()
    for link_idx in range(n_l_r, n_l_r + n_l_s):
        local_list = list()
        for flow_idx in range(n_f):
            if link_idx in link_pos_in_flows[flow_idx]:
                local_list += [
                    (flow_idx, pos) for pos in link_pos_in_flows[flow_idx][link_idx]
                ]
        path_to_s_link.append(local_list)

    # routers_groupings: two dimensional array, the first dimension indicates each of the routers, second dimension are the links
    # that start from that given router. Used later for grouping link states by source routers
    # routers_groupings_inversed: the same concept as routers_groupings but inversed. Single array with each position indicating
    # router links and its value indicate the router it comes from
    # Due to prunning, we know that each router will have at least one link originating from it
    routers_groupings = [list() for _ in range(n_r)]
    routers_groupings_inversed = list()
    for ii, r_link in enumerate(ordered_router_links):
        identified_router = router_id_mapping[r_link["node_id"]]
        routers_groupings[identified_router].append(ii)
        routers_groupings_inversed.append(identified_router)

    # switches_groupings: two dimensional array, the first dimension indicates each of the switches, second dimension are the links
    # that start from that given switch. Used later for grouping link states by source switches
    # switches_groupings_inversed: the same concept as switches_groupings but inversed. Single array with each position indicating
    # switch links and its value indicate the switch it comes from
    # Due to prunning, we know that each switch will have at least one link originating from it
    switches_groupings = [list() for _ in range(n_s)]
    switches_groupings_inversed = list()
    for ii, s_link in enumerate(ordered_switch_links):
        identified_switch = switch_id_mapping[s_link["node_id"]]
        switches_groupings[identified_switch].append(ii)
        switches_groupings_inversed.append(identified_switch)

    # Many of the features must have expanded dimensions so they can be concatenated
    sample = (
        {
            "sample_idx": sample_file_id,
            "sample_file_id": [sample_file_id] * n_f,
            "flow_id": [
                flow["flow_id"] for flow in ordered_cbr_flows + ordered_mb_flows
            ],
            # Agnostic attributes for all flows
            # Useful for RNN model
            "flow_traffic": np.expand_dims(
                [flow["traffic"] for flow in ordered_cbr_flows + ordered_mb_flows],
                axis=1,
            ),
            "flow_packets": np.expand_dims(
                [flow["packets"] for flow in ordered_cbr_flows + ordered_mb_flows],
                axis=1,
            ),
            "flow_packet_size": np.expand_dims(
                [flow["packet_size"] for flow in ordered_cbr_flows + ordered_mb_flows],
                axis=1,
            ),
            "flow_length": np.expand_dims(
                [flow["length"] for flow in ordered_cbr_flows + ordered_mb_flows],
                axis=1,
            ),
            "flow_packets_per_ms": tf.ragged.constant(
                [
                    flow["packets_per_ms"]
                    for flow in ordered_cbr_flows + ordered_mb_flows
                ],
                ragged_rank=1,
            ),
            "flow_delay": np.expand_dims(
                [flow["delay"] for flow in ordered_cbr_flows + ordered_mb_flows], axis=1
            ),
            # Link attributes
            "link_r_capacity": np.expand_dims(
                [link["capacity"] for link in ordered_router_links], axis=1
            ),
            "link_s_capacity": np.expand_dims(
                [link["capacity"] for link in ordered_switch_links], axis=1
            ),
            # Topology attributes
            "link_to_path": tf.ragged.constant(link_to_path),
            "path_to_r_link": tf.ragged.constant(path_to_r_link, ragged_rank=1),
            "path_to_s_link": tf.ragged.constant(path_to_s_link, ragged_rank=1),
            "routers_groupings": tf.ragged.constant(routers_groupings),
            "switches_groupings": tf.ragged.constant(switches_groupings),
            "routers_groupings_inversed": routers_groupings_inversed,
            "switches_groupings_inversed": switches_groupings_inversed,
        },
        np.expand_dims(
            [flow["delay"] for flow in ordered_cbr_flows + ordered_mb_flows], axis=1
        ),
    )

    return sample


def _generator(
    data_dir: str, shuffle: bool, verify_delays: bool
) -> Generator[Tuple[Dict[str, Any], List[float]], None, None]:
    """Returns processed samples from the given dataset.

    Parameters
    ----------
    data_dir : str
        Path to the dataset
    shuffle : bool
        True to shuffle the samples, False otherwise
    verify_delays: bool, optional
        True so that samples with unvalid delay values are discarded

    Yields
    ------
    Generator[Tuple[Dict[str, Any], List[float]], None, None]
        Returns a generator of tuples, where the first element is a dictionary with the sample's features
        and the second element is a list of the sample's labels (in this case, the flow's delay)
    """
    try:
        data_dir = data_dir.decode("UTF-8")
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    sample_idx = 0
    for sample in iter(tool):
        ret = _get_network_decomposition(sample, sample_idx)
        # SKIP SAMPLES WITH ZERO OR NEGATIVE VALUES
        if verify_delays and not all(x > 0 for x in ret[1]):
            continue
        sample_idx += 1
        yield ret


def input_fn(
    data_dir: str, shuffle: bool = False, verify_delays: bool = True
) -> tf.data.Dataset:
    """Returns a tf.data.Dataset object with the dataset stored in the given path

    Parameters
    ----------
    data_dir : str
        Path to the dataset
    shuffle : bool, optional
        True to shuffle the samples, False otherwise, by default False
    verify_delays: bool, optional
        True so that samples with unvalid delay values are discarded, by default True

    Returns
    -------
    tf.data.Dataset
        The processed dataset
    """
    signature = (
        {
            "sample_idx": tf.TensorSpec(shape=(), dtype=tf.int32),
            "sample_file_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "flow_id": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "flow_traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_packet_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_length": tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
            "flow_packets_per_ms": tf.RaggedTensorSpec(
                shape=(None, None, 1), dtype=tf.float32, ragged_rank=1
            ),
            "flow_delay": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_r_capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_s_capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_to_path": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
            "path_to_r_link": tf.RaggedTensorSpec(
                shape=(None, None, 2), dtype=tf.int32, ragged_rank=1
            ),
            "path_to_s_link": tf.RaggedTensorSpec(
                shape=(None, None, 2), dtype=tf.int32, ragged_rank=1
            ),
            "routers_groupings": tf.RaggedTensorSpec(
                shape=(None, None), dtype=tf.int32
            ),
            "switches_groupings": tf.RaggedTensorSpec(
                shape=(None, None), dtype=tf.int32
            ),
            "routers_groupings_inversed": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "switches_groupings_inversed": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        _generator,
        args=[data_dir, shuffle, verify_delays],
        output_signature=signature,
    )

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


# MAIN: generate the dataset

# Set seeds for reproducibility
np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

# Parse dataset
tf.data.Dataset.save(
    input_fn(
        args.input_dir,
        shuffle=args.shuffle and not args.test,
        verify_delays=not args.test,
    ),
    args.output_dir,
    compression="GZIP",
)

if not args.test:

    # 80/20 training/validation split
    print("Splitting dataset into training and validation")
    ds = tf.data.Dataset.load(args.output_dir, compression="GZIP")
    val_size = int(0.2 * len(ds))
    tf.data.Dataset.save(
        ds.take(val_size),
        os.path.join(args.output_dir, "validation"),
        compression="GZIP",
    )
    tf.data.Dataset.save(
        ds.skip(val_size), os.path.join(args.output_dir, "training"), compression="GZIP"
    )
