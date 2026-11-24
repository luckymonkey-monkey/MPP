from pathlib import Path
import tempfile
import unittest

from mpp.data import load_records


def write_dataset(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class DataTests(unittest.TestCase):
    def test_load_records_deduplicates_equal_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_dataset(Path(directory) / "data.txt", "CCO 10\nCCO 10\nCCC 20\n")
            records = load_records(path)
        self.assertEqual(
            [(record.smiles, record.pce) for record in records],
            [("CCO", 10.0), ("CCC", 20.0)],
        )

    def test_load_records_rejects_conflicting_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_dataset(Path(directory) / "data.txt", "CCO 10\nCCO 11\n")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                load_records(path)

    def test_load_records_rejects_invalid_rows(self) -> None:
        cases = [
            ("CCO\n", "expected"),
            ("CCO nope\n", "numeric"),
            ("CCO 101\n", "between 0 and 100"),
            ("CCO.CN 20\n", "multi-fragment"),
            ("not-a-smiles 20\n", "RDKit rejected"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            for text, message in cases:
                with self.subTest(text=text):
                    write_dataset(path, text)
                    with self.assertRaisesRegex(ValueError, message):
                        load_records(path)


if __name__ == "__main__":
    unittest.main()
