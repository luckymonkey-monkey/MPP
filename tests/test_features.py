import unittest

from mpp.features import GraphFeaturizer, UnknownFeatureError


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.featurizer = GraphFeaturizer(radius=1).fit(["CCO"])

    def test_featurizer_produces_matching_graph_shapes(self) -> None:
        sample = self.featurizer.transform("CCO")
        self.assertEqual(sample.fingerprints.ndim, 1)
        self.assertEqual(sample.adjacency.shape, (sample.n_atoms, sample.n_atoms))
        self.assertEqual(sample.unknown_features, 0)

    def test_featurizer_rejects_unknown_features_in_strict_mode(self) -> None:
        with self.assertRaises(UnknownFeatureError):
            self.featurizer.transform("CCl")

    def test_featurizer_can_count_unknown_features(self) -> None:
        sample = self.featurizer.transform("CCl", unknown_policy="zero")
        self.assertGreater(sample.unknown_features, 0)

    def test_featurizer_rejects_empty_and_fragmented_smiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.featurizer.transform("")
        with self.assertRaisesRegex(ValueError, "Multi-fragment"):
            self.featurizer.transform("CCO.CN")


if __name__ == "__main__":
    unittest.main()
