from .flows import Flow


class FlowGenerator:
    """Generator for Flows from packets extraced using reader.Reader.read()"""

    def combine(self, packets):
        """Combine individual packets into a flow representation

            Parameters
            ----------
            packets : np.array of shape=(n_samples_packets, n_features_packets)
                Output from Reader.read

            Returns
            -------
            flows : dict
                Dictionary of flow_key -> Flow()
            """
        result = dict()

        for packet in packets:
            key = (packet[0], packet[1], packet[2])
            result[key] = result.get(key, Flow()).add(packet)

        return result
