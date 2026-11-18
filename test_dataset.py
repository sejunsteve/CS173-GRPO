import copy
import tempfile
import unittest
from pathlib import Path
from dataset import generate, validate, audit_rows, write_csv, load_rows
from training import training_records


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.attempts = generate(173)

    def test_quotas_and_reproducibility(self):
        summary = validate(self.splits)
        self.assertEqual(len(self.splits['train']), 1040)
        self.assertEqual(len(self.splits['validation']), 260)
        self.assertEqual(len(self.splits['test']), 260)
        again, attempts = generate(173)
        self.assertEqual(self.splits, again)
        self.assertEqual(self.attempts, attempts)

    def test_audit_train_only_and_unverified(self):
        rows = audit_rows(self.splits['train'])
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r['example_id'] for r in rows}), 20)
        self.assertEqual(len({r['expected_rule_id'] for r in rows}), 13)
        self.assertTrue(all(r['split'] == 'train' and r['human_verified'] == '' for r in rows))

    def test_csv_roundtrip_and_no_answer_in_model_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.csv'
            write_csv(path, self.splits['train'])
            self.assertEqual(len(load_rows(path)), 1040)
            records = training_records(path)
        self.assertEqual(set(records[0]), {'prompt', 'request_json'})
        self.assertEqual(records[0]['prompt'][0]['role'], 'system')
        self.assertEqual(records[0]['prompt'][-1]['role'], 'user')

    def test_corrupted_answer_detected(self):
        bad = copy.deepcopy(self.splits)
        bad['train'][0]['expected_output'] = '{"decision":"ALLOW","rule_id":"R99"}'
        with self.assertRaises((ValueError, AssertionError)):
            validate(bad)

    def test_cross_split_duplicate_detected(self):
        bad = copy.deepcopy(self.splits)
        bad['test'][0] = dict(bad['train'][0], split='test')
        with self.assertRaises((ValueError, AssertionError)):
            validate(bad)


if __name__ == '__main__':
    unittest.main()
