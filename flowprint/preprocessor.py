import pickle
import sys

import numpy as np

from .flow_generator import FlowGenerator
from .reader import Reader


class Preprocessor:
    """Preprocessor object for preprocessing flows from pcap files

        Attributes
        ----------
        reader : reader.Reader
            pcap Reader object for reading .pcap files

        flow_generator : flows.FlowGenerator
            Flow generator object for generating Flow objects
    """

    def __init__(self, verbose=False):
        """Preprocessor object for preprocessing flows from pcap files"""
        self.reader = Reader(verbose)
        self.flow_generator = FlowGenerator()

    def process(self, files, labels):
        """Extract data from files and attach given labels.

            Parameters
            ----------
            files : iterable of string
                Paths from which to extract data.

            labels : iterable of int
                Label corresponding to each path.

            Returns
            -------
            X : np.array of shape=(n_samples, n_features)
                Features extracted from files.

            y : np.array of shape=(n_samples,)
                Labels for each flow extracted from files.
            """
        # Initialise X and y
        X, y = list(), list()

        for file, label in zip(files, labels):
            try:
                data = np.array(list(self.extract(file).values()))
            except KeyboardInterrupt:
                break
            except Exception as ex:
                print(f"Reading {file} failed: '{ex}'", file=sys.stderr)
                continue
            X.append(data)
            y.append(np.array([label] * data.shape[0]))

        # Filter empty entries from array
        X = list(filter(lambda x: x.shape[0] != 0, X))
        y = list(filter(lambda x: x.shape[0] != 0, y))

        try:
            X = np.concatenate(X)
            y = np.concatenate(y)
        except Exception:
            X = np.array([], dtype=object)
            y = np.array([], dtype=object)

        return X, y

    def extract(self, infile):
        """Extract flows from given pcap file.

            Parameters
            ----------
            infile : string
                Path to input file.

            Returns
            -------
            result : dict
                Dictionary of flow_key -> flow.
            """
        # Read packets
        result = self.reader.read(infile)
        # Combine packets into flows
        result = self.flow_generator.combine(result)
        return result

    def save(self, outfile, X, y):
        """Save data to given outfile.

            Parameters
            ----------
            outfile : string
                Path of file to save data to.

            X : np.array of shape=(n_samples, n_features)
                Features extracted from files.

            y : np.array of shape=(n_samples,)
                Labels for each flow extracted from files.
            """
        with open(outfile, "wb") as outfile:
            pickle.dump((X, y), outfile)

    def load(self, infile):
        """Load data from given infile.

            Parameters
            ----------
            infile : string
                Path of file from which to load data.

            Returns
            -------
            X : np.array of shape=(n_samples, n_features)
                Features extracted from files.

            y : np.array of shape=(n_samples,)
                Labels for each flow extracted from files.
            """
        with open(infile, "rb") as infile:
            return pickle.load(infile)
