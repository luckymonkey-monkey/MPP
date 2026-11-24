import torch
import unittest

from mpp.features import GraphFeaturizer
from mpp.model import MPPConfig, MPPRegressor, collate_graphs


class ModelTests(unittest.TestCase):
    def test_model_returns_one_value_per_molecule(self) -> None:
        featurizer = GraphFeaturizer(radius=1).fit(["CCO", "CCC"])
        samples = [
            featurizer.transform("CCO", pce=10.0),
            featurizer.transform("CCC", pce=20.0),
        ]
        config = MPPConfig(n_fingerprints=len(featurizer.vocabulary.fingerprints))
        model = MPPRegressor(config)
        batch = collate_graphs(samples, torch.device("cpu"))
        output = model(batch[0], batch[1], batch[2])
        self.assertEqual(output.shape, (2,))
        self.assertEqual(batch[3].shape, (2,))

    def test_model_rejects_inconsistent_sizes(self) -> None:
        model = MPPRegressor(MPPConfig(n_fingerprints=2))
        with self.assertRaisesRegex(ValueError, "sum"):
            model(torch.tensor([0, 1]), torch.eye(2), [1])


if __name__ == "__main__":
    unittest.main()
