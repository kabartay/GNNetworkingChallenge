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

import tensorflow as tf
import tensorflow_probability as tfp


class Baseline_cbr_mb(tf.keras.Model):
    min_max_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
    }
    min_max_scores = None

    name = "Baseline_cbr_mb"

    def __init__(self, override_min_max_scores=None, name=None):
        super(Baseline_cbr_mb, self).__init__()

        self.iterations = 8
        self.path_state_dim = 64
        self.link_state_dim = 64

        if override_min_max_scores is not None:
            self.set_min_max_scores(override_min_max_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate"),
            return_sequences=True,
            return_state=True,
            name="PathUpdateRNN",
        )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )

        self.flow_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=5),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="PathEmbedding",
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=2),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="LinkEmbedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
            ],
            name="PathReadout",
        )

    def set_min_max_scores(self, override_min_max_scores):
        assert (
            type(override_min_max_scores) == dict
            and all(kk in override_min_max_scores for kk in self.min_max_scores_fields)
            and all(len(val) == 2 for val in override_min_max_scores.values())
        ), "overriden min-max dict is not valid!"
        self.min_max_scores = override_min_max_scores

    @tf.function
    def call(self, inputs):
        # Ensure that the min-max scores are set
        assert (
            self.min_max_scores is not None
        ), "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_packets = inputs["flow_packets"]
        flow_packet_size = inputs["flow_packet_size"]
        flow_type = inputs["flow_type"]
        link_capacity = inputs["link_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]

        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)

        # Initialize the initial hidden state for paths
        path_state = self.flow_embedding(
            tf.concat(
                [
                    (flow_traffic - self.min_max_scores["flow_traffic"][0])
                    * self.min_max_scores["flow_traffic"][1],
                    (flow_packets - self.min_max_scores["flow_packets"][0])
                    * self.min_max_scores["flow_packets"][1],
                    (flow_packet_size - self.min_max_scores["flow_packet_size"][0])
                    * self.min_max_scores["flow_packet_size"][1],
                    flow_type,
                ],
                axis=1,
            )
        )

        # Initialize the initial hidden state for links
        link_state = self.link_embedding(
            tf.concat(
                [
                    (link_capacity - self.min_max_scores["link_capacity"][0])
                    * self.min_max_scores["link_capacity"][1],
                    load,
                ],
                axis=1,
            ),
        )

        # Iterate t times doing the message passing
        for _ in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            previous_path_state = path_state
            path_state_sequence, path_state = self.path_update(
                link_gather, initial_state=path_state
            )
            # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #   PATH TO LINK  #
            ###################
            path_gather = tf.gather_nd(
                path_state_sequence, path_to_link, name="PathToRLink"
            )
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            link_state, _ = self.link_update(path_sum, states=link_state)

        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])
        capacity_gather = tf.gather(link_capacity, link_to_path)
        delay_sequence = occupancy / capacity_gather
        delay = tf.math.reduce_sum(delay_sequence, axis=1)
        return delay


class Baseline_mb(tf.keras.Model):
    min_max_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
    }

    name = "Baseline_mb"

    def __init__(self, override_min_max_scores=None, name=None):
        super(Baseline_mb, self).__init__()

        self.iterations = 8
        self.path_state_dim = 64
        self.link_state_dim = 64

        if override_min_max_scores is not None:
            self.set_min_max_scores(override_min_max_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate"),
            return_sequences=True,
            return_state=True,
            name="PathUpdateRNN",
        )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )

        self.path_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=3),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="PathEmbedding",
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=2),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="LinkEmbedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
            ],
            name="PathReadout",
        )

    def set_min_max_scores(self, override_min_max_scores):
        assert (
            type(override_min_max_scores) == dict
            and all(kk in override_min_max_scores for kk in self.min_max_scores_fields)
            and all(len(val) == 2 for val in override_min_max_scores.values())
        ), "overriden min-max dict is not valid!"
        self.min_max_scores = override_min_max_scores

    @tf.function
    def call(self, inputs):
        # Ensure that the min-max scores are set
        assert (
            self.min_max_scores is not None
        ), "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_packets = inputs["flow_packets"]
        flow_packet_size = inputs["flow_packet_size"]
        link_capacity = inputs["link_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]

        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)

        # Initialize the initial hidden state for paths
        path_state = self.path_embedding(
            tf.concat(
                [
                    (flow_traffic - self.min_max_scores["flow_traffic"][0])
                    * self.min_max_scores["flow_traffic"][1],
                    (flow_packets - self.min_max_scores["flow_packets"][0])
                    * self.min_max_scores["flow_packets"][1],
                    (flow_packet_size - self.min_max_scores["flow_packet_size"][0])
                    * self.min_max_scores["flow_packet_size"][1],
                ],
                axis=1,
            )
        )

        # Initialize the initial hidden state for links
        link_state = self.link_embedding(
            tf.concat(
                [
                    (link_capacity - self.min_max_scores["link_capacity"][0])
                    * self.min_max_scores["link_capacity"][1],
                    load,
                ],
                axis=1,
            ),
        )

        # Iterate t times doing the message passing
        for _ in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            previous_path_state = path_state
            path_state_sequence, path_state = self.path_update(
                link_gather, initial_state=path_state
            )
            # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #   PATH TO LINK  #
            ###################
            path_gather = tf.gather_nd(
                path_state_sequence, path_to_link, name="PathToRLink"
            )
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            link_state, _ = self.link_update(path_sum, states=link_state)

        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])
        capacity_gather = tf.gather(link_capacity, link_to_path)
        delay_sequence = occupancy / capacity_gather
        delay = tf.math.reduce_sum(delay_sequence, axis=1)
        return delay


class GNN_proposed(tf.keras.Model):
    min_max_scores_fields = {"flow_traffic", "flow_packets", "flow_packet_size"}
    name = "GNN_proposed"

    def __init__(
        self,
        flow_state_dim=64,
        link_state_dim=64,
        node_state_dim=16,
        threshold=0.05,
        max_iterations=10,  # 40
        l1=0,
        l2=0,
        dropout=0,
        override_min_max_scores=None,
        name=None,
        log=False,
    ):
        super(GNN_proposed, self).__init__()

        self.max_iterations = max_iterations
        self.threshold = threshold

        self.flow_state_dim = flow_state_dim
        self.link_state_dim = link_state_dim
        self.node_state_dim = node_state_dim

        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout

        if override_min_max_scores is not None:
            self.set_min_max_scores(override_min_max_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        assert type(log) == bool, "log must be a boolean"
        self.log = log

        # GRU Cells used in the Message Passing step
        self.flow_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.flow_state_dim, name="FlowUpdate"),
            return_sequences=True,
            return_state=True,
            name="FlowUpdateRNN",
        )
        self.link_r_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="RouterLinkUpdate"
        )
        self.link_s_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="SwitchLinkUpdate"
        )
        self.router_update = tf.keras.layers.GRUCell(
            self.node_state_dim, name="RouterUpdate"
        )
        self.switch_update = tf.keras.layers.GRUCell(
            self.node_state_dim, name="SwitchUpdate"
        )
        # GRU Cell for encoding packets per ms
        self.flow_encoder = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.flow_state_dim, name="FlowEncoder"),
            return_sequences=True,
            return_state=True,
            name="FlowEncoderRNN",
        )

        # Embedding layers
        self.flow_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=3 + self.flow_state_dim),
                tf.keras.layers.Dense(
                    self.flow_state_dim,
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2),
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(
                    self.flow_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="FlowEmbedding",
        )

        self.link_r_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=1),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="RouterLinkEmbedding",
        )
        self.link_s_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=1),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="SwitchLinkEmbedding",
        )
        self.router_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.link_state_dim),
                tf.keras.layers.Dense(
                    (self.link_state_dim + self.node_state_dim) // 2,
                    activation=tf.keras.activations.relu,
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(
                    self.node_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )
        self.switch_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.link_state_dim),
                tf.keras.layers.Dense(
                    (self.link_state_dim + self.node_state_dim) // 2,
                    activation=tf.keras.activations.relu,
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(
                    self.node_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.readout_flow = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.flow_state_dim),
                tf.keras.layers.Dense(
                    self.flow_state_dim // 2,
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2),
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(
                    self.flow_state_dim // 4,
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2),
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
            ],
            name="FlowReadout",
        )

    def set_min_max_scores(self, override_min_max_scores):
        assert (
            type(override_min_max_scores) == dict
            and all(kk in override_min_max_scores for kk in self.min_max_scores_fields)
            and all(len(val) == 2 for val in override_min_max_scores.values())
        ), "overriden min-max dict is not valid!"
        self.min_max_scores = override_min_max_scores

    @tf.function
    def condition(
        self,
        flow_state_sequence,
        initial_flow_state,
        flow_state,
        previous_flow_state,
        link_s_state,
        link_r_state,
        router_state,
        switch_state,
        path_to_s_link,
        path_to_r_link,
        link_to_path,
        routers_groupings,
        switches_groupings,
        routers_groupings_inversed,
        switches_groupings_inversed,
        ii,
    ):
        flow_err_mean = tf.reduce_mean(
            tf.abs((flow_state - previous_flow_state) / (previous_flow_state + 1e-9)),
            axis=1,
        )
        flow_percentile = tfp.stats.percentile(flow_err_mean, 95)
        c_flow = tf.less(flow_percentile, self.threshold)
        c_iter = tf.greater_equal(ii, self.max_iterations)
        return tf.logical_not(tf.logical_or(c_iter, c_flow))

    @tf.function
    def message_passing(
        self,
        flow_state_sequence,
        initial_flow_state,
        flow_state,
        previous_flow_state,
        link_s_state,
        link_r_state,
        router_state,
        switch_state,
        path_to_s_link,
        path_to_r_link,
        link_to_path,
        routers_groupings,
        switches_groupings,
        routers_groupings_inversed,
        switches_groupings_inversed,
        ii,
    ):
        previous_flow_state = flow_state
        ####################
        #  LINKS TO PATH   #
        ####################
        routers_nodes_and_links_states = tf.concat(
            [
                link_r_state,
                tf.gather(
                    router_state,
                    routers_groupings_inversed,
                    name="RStateUnfolded",
                ),
            ],
            axis=1,
        )
        switches_nodes_and_links_states = tf.concat(
            [
                link_s_state,
                tf.gather(
                    switch_state,
                    switches_groupings_inversed,
                    name="SStateUnfolded",
                ),
            ],
            axis=1,
        )
        combined_link_state = tf.concat(
            [routers_nodes_and_links_states, switches_nodes_and_links_states],
            axis=0,
        )
        link_gather = tf.gather(combined_link_state, link_to_path, name="LinkToPath")

        flow_state_sequence, flow_state = self.flow_update(
            link_gather, initial_state=initial_flow_state
        )
        # We select the element in flow_state_sequence so that it corresponds to the
        # state before the link was considered
        flow_state_sequence = tf.concat(
            [tf.expand_dims(previous_flow_state, 1), flow_state_sequence], axis=1
        )

        ###################
        #  PATH TO ROUTER #
        #      LINK       #
        ###################
        flow_gather = tf.gather_nd(
            flow_state_sequence, path_to_r_link, name="FlowToRLink"
        )
        flow_sum = tf.math.reduce_sum(flow_gather, axis=1)
        link_r_state, _ = self.link_r_update(flow_sum, states=link_r_state)
        ###################
        #  ROUTER LINK TO #
        #      ROUTER     #
        ###################
        router_state, _ = self.router_update(
            tf.math.reduce_sum(
                tf.gather(link_r_state, routers_groupings),
                axis=1,
                name="RLinkGrouping",
            ),
            states=router_state,
        )

        ###################
        #  PATH TO SWITCH #
        #      LINK       #
        ###################
        flow_gather = tf.gather_nd(
            flow_state_sequence, path_to_s_link, name="FlowToSLink"
        )
        flow_sum = tf.math.reduce_sum(flow_gather, axis=1)
        link_s_state, _ = self.link_s_update(flow_sum, states=link_s_state)
        ###################
        #  SWITCH LINK TO #
        #      SWITCH     #
        ###################
        switch_state, _ = self.switch_update(
            tf.math.reduce_sum(
                tf.gather(link_s_state, switches_groupings),
                axis=1,
                name="SLinkGrouping",
            ),
            states=switch_state,
        )

        # Complete iteration
        ii = ii + 1
        return (
            flow_state_sequence,
            initial_flow_state,
            flow_state,
            previous_flow_state,
            link_s_state,
            link_r_state,
            router_state,
            switch_state,
            path_to_s_link,
            path_to_r_link,
            link_to_path,
            routers_groupings,
            switches_groupings,
            routers_groupings_inversed,
            switches_groupings_inversed,
            ii,
        )

    @tf.function
    def call(self, inputs, training=None):

        if self.min_max_scores is None:
            raise NotImplementedError(
                "default min-max values haven't been implemented!"
            )

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_packets = inputs["flow_packets"]
        flow_packet_size = inputs["flow_packet_size"]
        flow_packets_per_ms = inputs["flow_packets_per_ms"]
        link_r_capacity = inputs["link_r_capacity"]
        link_s_capacity = inputs["link_s_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_r_link = inputs["path_to_r_link"]
        path_to_s_link = inputs["path_to_s_link"]
        flow_length = inputs["flow_length"]
        routers_groupings = inputs["routers_groupings"]
        switches_groupings = inputs["switches_groupings"]
        routers_groupings_inversed = inputs["routers_groupings_inversed"]
        switches_groupings_inversed = inputs["switches_groupings_inversed"]

        # Encode flow_packets_per_ms
        _, encoded_traffic = self.flow_encoder(flow_packets_per_ms)

        # Obtain load in links
        flow_gather_traffic_r = tf.gather(flow_traffic, path_to_r_link[:, :, 0])
        flow_gather_traffic_s = tf.gather(flow_traffic, path_to_s_link[:, :, 0])
        load_r = tf.math.reduce_sum(flow_gather_traffic_r, axis=1) / (
            link_r_capacity * 1e9
        )
        load_s = tf.math.reduce_sum(flow_gather_traffic_s, axis=1) / (
            link_s_capacity * 1e9
        )

        # Initialize the initial hidden state for flows
        flow_state = self.flow_embedding(
            tf.concat(
                [
                    (flow_traffic - self.min_max_scores["flow_traffic"][0])
                    * self.min_max_scores["flow_traffic"][1],
                    (flow_packets - self.min_max_scores["flow_packets"][0])
                    * self.min_max_scores["flow_packets"][1],
                    (flow_packet_size - self.min_max_scores["flow_packet_size"][0])
                    * self.min_max_scores["flow_packet_size"][1],
                    encoded_traffic,
                ],
                axis=1,
            )
        )

        # Initialize the initial hidden state for links
        link_r_state = self.link_r_embedding(load_r)
        link_s_state = self.link_s_embedding(load_s)

        # Initialize the initial hidden state for nodes
        router_state = self.router_embedding(
            tf.math.reduce_sum(
                tf.gather(link_r_state, routers_groupings),
                axis=1,
                name="RLinkGrouping-Embedding",
            ),
        )
        switch_state = self.switch_embedding(
            tf.math.reduce_sum(
                tf.gather(link_s_state, switches_groupings),
                axis=1,
                name="SLinkGrouping-Embedding",
            ),
        )

        # Iterate t times doing the message passing
        # first message passing is called manually to initialize the flow_state_sequence
        ii = tf.constant(0)
        flow_state_sequence = tf.while_loop(
            self.condition,
            self.message_passing,
            self.message_passing(
                None,
                flow_state,
                flow_state,
                None,
                link_s_state,
                link_r_state,
                router_state,
                switch_state,
                path_to_s_link,
                path_to_r_link,
                link_to_path,
                routers_groupings,
                switches_groupings,
                routers_groupings_inversed,
                switches_groupings_inversed,
                ii,
            ),
        )[0]

        ################
        #  READOUT     #
        ################
        occupancy = self.readout_flow(flow_state_sequence[:, 1:])
        capacity = tf.concat([link_r_capacity, link_s_capacity], axis=0)
        capacity_gather = tf.gather(capacity, link_to_path)
        delay_sequence = occupancy / capacity_gather
        queuing_delay = tf.math.reduce_sum(delay_sequence, axis=1)
        # We assume fixed propagation delay of 1ns per link
        prop_delay = tf.cast(flow_length, tf.float32) * 1e-6
        transmision_delay = tf.reduce_sum(
            tf.expand_dims(flow_packet_size, 1)
            / (tf.math.reduce_sum(capacity_gather, axis=1) * 1e9),
            axis=1,
        )
        delay = queuing_delay + prop_delay + transmision_delay

        if self.log:
            return tf.math.log(delay)
        return delay
