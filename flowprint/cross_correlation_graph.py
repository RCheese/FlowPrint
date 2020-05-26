from collections import Counter
from itertools import combinations
import json
import warnings

import networkx as nx


class CrossCorrelationGraph:
    """CrossCorrelationGraph for computing correlation between clusters

        Attributes
        ----------
        window : float
            Threshold for the window size in seconds

        correlation : float
            Threshold for the minimum required correlation

        graph : nx.Graph
            Cross correlation graph containing all correlations
            Note that each node in the graph represents an 'activity signature'
            to avoid duplicates. The NetworkDestinations corresponding to each
            signature are stored in the 'mapping' attribute.

            Note
            ----
            IMPORTANT: The CrossCorrelation.graph object is an optimised graph.
            Each node does not represent a network destination, but represents
            an activity fingerprint. E.g. when destinations A and B are both
            only active at time slices 3 and 7, then these destinations are
            represented by a single node. We use the self.mapping to extract the
            network destinations from each graph node.
            This is a huge optimisation for finding cliques as the number of
            different network destinations theoretically covers the entire IP
            space, whereas the number of activity fingerprints is bounded by
            2^(batch / window), in our work 2^(300/30) = 2^10 = 1024. If these
            parameters change, the complexity may increase, but never beyond the
            original bounds. Hence, this optimisation never has a worse time
            complexity.

        mapping : dict
            NetworkDestinations corresponding to each node in the graph
    """

    def __init__(self, window=30, correlation=0.1):
        """CrossCorrelationGraph for computing correlation between clusters

            Parameters
            ----------
            window : float, default=30
                Threshold for the window size in seconds

            correlation : float, default=0.1
                Threshold for the minimum required correlation
            """
        # Set parameters
        self.window = window
        self.correlation = correlation
        self.mapping = dict()
        self.graph = nx.Graph()

    def fit(self, cluster, y=None):
        """Fit Cross Correlation Graph.

            Parameters
            ----------
            cluster : Cluster
                Cluster to fit graph, cluster must be populated with flows

            y : ignored

            Returns
            -------
            result : self
                Returns self
            """
        # Compute cross correlations within cluster
        correlations, self.mapping = self.cross_correlation(cluster)

        if self.correlation <= 0:  # Create a fully connected graph
            self.graph = nx.complete_graph(list(self.mapping.keys()))
        else:
            self.graph = nx.Graph()
            self.graph.add_nodes_from(list(self.mapping.keys()))
            for (u, v), weight in correlations.items():
                if weight >= self.correlation:
                    self.graph.add_edge(u, v, weight=weight)
        return self

    def predict(self, X=None, y=None):
        """Fit Cross Correlation Graph and return cliques.

            Parameters
            ----------
            X : ignored

            y : ignored

            Returns
            -------
            result : Generator of cliques
                Generator of all cliques in the graph
            """
        cliques = nx.find_cliques(self.graph)
        return (set.union(*[self.mapping.get(n) for n in c]) for c in cliques)

    def fit_predict(self, cluster, y=None):
        """Fit cross correlation graph with clusters from X and return cliques.

            Parameters
            ----------
            cluster : Cluster
                Cluster to fit graph, cluster must be populated with flows

            y : ignored

            Returns
            -------
            result : Generator of cliques
                Generator of all cliques in the graph
            """
        return self.fit(cluster).predict(cluster)

    def export(self, outfile, dense=True, format="gexf"):
        """Export CrossCorrelationGraph to outfile for further analysis

            Parameters
            ----------
            outfile : string
                File to export CrossCorrelationGraph

            dense : boolean, default=True
                If True  export the dense graph (see IMPORTANT note at graph),
                this means that each node is represented by the time slices in
                which they were active. Each node still has the information of
                all correlated nodes.

                If False export the complete graph. Note that these graphs can
                get very large with lots of edges, therefore, for manual
                inspection it is recommended to use dense=True instead.

            format : ('gexf'|'gml'), default='gexf'
                Format in which to export, currently only 'gexf', 'gml' are
                supported.
            """
        if dense:
            graph = self.graph

            # Initialise human-readable mapping of nodes
            mapping = dict()
            # Fill mapping
            for node in graph:
                info = {
                    "window": list(sorted(node)),
                    "ips": set(),
                    "certs": set(),
                    "labels": Counter(),
                }
                # Loop over corresponding network destinations
                for destination in self.mapping.get(node):
                    info["ips"] = info.get("ips", set()) | destination.destinations
                    info["certs"] = info.get("certs", set()) | destination.certificates
                    info["labels"] = info.get("labels", Counter()) + destination.labels

                # Remove None from certificates
                info["certs"] = info.get("certs", set()) - {None}
                # Transform sets into lists
                info["ips"] = list(info.get("ips", set()))
                info["certs"] = list(info.get("certs", set()))

                # Store mapping as text
                mapping[node] = json.dumps(info, sort_keys=True)

            graph = nx.relabel_nodes(graph, mapping)

        # Make graph not dense
        else:
            graph = nx.Graph()
            for node in self.graph:
                for destination in self.mapping.get(node):
                    graph.add_node(destination)

            for node in self.graph:
                for source in self.mapping.get(node):
                    for destination in self.mapping.get(node):
                        if source == destination:
                            continue
                        graph.add_edge(source, destination, weight=1)

                    # Add all edges to other nodes
                    for connected in nx.neighbors(self.graph, node):
                        # Get edge get_edge_data
                        data = self.graph.get_edge_data(node, connected)
                        # Get all destinations
                        for destination in self.mapping.get(connected):
                            graph.add_edge(source, destination, data=data)

            # Transform network destinations to human readable format
            mapping = dict()
            for node in self.graph:
                for destination in self.mapping.get(node):
                    info = {
                        "window": list(sorted(node)),
                        "ips": list(destination.destinations),
                        "certs": list(destination.certificates - {None}),
                        "labels": destination.labels,
                    }

                    mapping[destination] = json.dumps(info, sort_keys=True)

            graph = nx.relabel_nodes(graph, mapping)

        # Export graph to file
        if format.lower() == "gexf":
            nx.write_gexf(graph, outfile)
        elif format.lower() == "gml":
            nx.write_gml(graph, outfile)
        else:
            # Warn user of unknown format
            warnings.warn(f"Unknown export format '{format}', defaulting to 'gexf'")
            # Export as gexf
            nx.write_gexf(graph, outfile)

    def cross_correlation(self, cluster):
        """Compute cross correlation between clusters

            Parameters
            ----------
            cluster : Cluster
                Cluster to fit graph, cluster must be populated with flows

            Returns
            -------
            correlation : dict
                Dictionary of cross correlation values between each
                NetworkDestination inside cluster.

            mapping : dict
                Mapping of activity fingerprint -> clusters
            """
        correlation = dict()

        # Get activity of samples
        activity = self.activity(cluster)
        # Get inverted mapping
        mapping = dict()
        for destination, active in activity.items():
            mapping[frozenset(active)] = mapping.get(frozenset(active), set()) | set(
                [destination]
            )

        # Compute cross correlation values
        for x, y in combinations(mapping, 2):
            union = len(x & y)
            if union:
                intersection = len(x | y)
                correlation[x, y] = union / intersection

        return correlation, mapping

    def activity(self, cluster):
        """Extracts sets of active clusters by time.

            Parameters
            ----------
            cluster : Cluster
                Cluster to fit graph, cluster must be populated with flows

            Returns
            -------
            mapping : dict
                Dictionary of NetworkDestination -> activity
            """
        X = cluster.samples
        start = min(x.time_start for x in X)

        # Initialise mapping of NetworkDestination -> activity
        mapping = dict()

        for destination in cluster.clusters():
            for flow in destination.samples:
                activity = set()
                for timestamp in flow.timestamps:
                    activity.add(int((timestamp - start) // self.window))
                mapping[destination] = mapping.get(destination, set()) | activity

        return mapping
