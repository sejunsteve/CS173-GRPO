# tests/test_rcac_audit.py
import unittest, json, tempfile, copy
from pathlib import Path

from policy import evaluate, make_prompt, canonical, RULE_IDS
from rewards import parse_output, access_reward, score, completion_text
from dataset import generate, validate, load_rows, write_csv


class TestPolicyAndRewards(unittest.TestCase):
    def setUp(self):
        # a minimal request hitting R06 (admin, emergency, incident, read)
        self.req_emergency = {
            "role": "admin", "user_dept": "ENG", "resource_dept": "ENG",
            "clearance": 3, "classification": 2,
            "resource_state": "active", "action": "read",
            "is_owner": False, "is_delegate": False,
            "mfa": False, "emergency": True,
            "network": "corporate", "hour": 10, "purpose": "incident"
        }
        # a request that should be DENY by R02 (clearance < classification)
        self.req_low_clear = {
            "role": "viewer", "user_dept": "FIN", "resource_dept": "FIN",
            "clearance": 0, "classification": 2,
            "resource_state": "active", "action": "read",
            "is_owner": False, "is_delegate": False,
            "mfa": True, "emergency": False,
            "network": "corporate", "hour": 12, "purpose": "routine"
        }

    def test_reward_batch_heterogeneous(self):
        """Two different requests, one completion per request in this callback batch."""
        completions = [
            [{"role": "assistant", "content": json.dumps(
                {"decision": "ALLOW", "rule_id": "R06"})}],
            [{"role": "assistant", "content": json.dumps(
                {"decision": "DENY", "rule_id": "R02"})}]
        ]
        request_json = [json.dumps(self.req_emergency), json.dumps(self.req_low_clear)]
        rewards = access_reward(completions, request_json)
        self.assertEqual(len(rewards), 2)
        # first reward should be perfect (decision+rule match)
        self.assertAlmostEqual(rewards[0], 1.0)
        # Both decision and rule are correct for the second request as well.
        self.assertAlmostEqual(rewards[1], 1.0)

    def test_emergency_priority_edge(self):
        """R04 precedes the emergency override R06 and must deny this request."""
        req = self.req_emergency.copy()
        req.update({"network": "public", "purpose": "incident"})
        ev = evaluate(req)
        self.assertEqual(ev["decision"], "DENY")
        self.assertEqual(ev["rule_id"], "R04")

    def test_request_json_tamper_detection(self):
        """Changing a feature after canonicalisation must change the decision."""
        orig = canonical(self.req_low_clear)
        tampered = json.loads(orig)
        tampered["clearance"] = 3  # now clearance >= classification
        # parse via policy (should be ALLOW by later rules, not DENY)
        ev_orig = evaluate(self.req_low_clear)
        ev_tampered = evaluate(tampered)
        self.assertNotEqual(ev_orig["decision"], ev_tampered["decision"])
        # reward must reflect the tampered request when fed to access_reward
        comp = [{"role": "assistant", "content": json.dumps(
                {"decision": ev_tampered["decision"], "rule_id": ev_tampered["rule_id"]})}]
        rewards = access_reward([comp], [json.dumps(tampered)])
        self.assertGreaterEqual(rewards[0], 0.4)  # at least decision correct

    def test_generated_output_corruption(self):
        """parse_output must reject malformed JSON or extra keys."""
        bad_texts = [
            '{"decision":"ALLOW","rule_id":"R06","extra":"no"}',          # extra key
            '{"decision":"ALLOW"}',                                         # missing rule_id
            'Just some text',                                                # not JSON
            '[{"decision":"ALLOW","rule_id":"R06"}]',                       # array not object
            '{"decision":123,"rule_id":"R06"}'                              # wrong type
        ]
        for txt in bad_texts:
            self.assertIsNone(parse_output(txt))

    def test_dataset_quota_and_validation(self):
        """Generate a tiny dataset (seeded) and verify quota & validation logic."""
        splits, attempts = generate(seed=42)  # uses the real generator (may be large)
        # sanity‑check that each split respects the per‑rule quotas
        summary = validate(splits)
        self.assertTrue(summary["ok"], msg="Dataset validation failed: " + "; ".join(summary["issues"]))
        # ensure that the total number of examples equals sum of quotas * rule count
        expected_total = sum(120 for _ in RULE_IDS)  # 80+20+20 per rule
        self.assertEqual(summary["total"], expected_total)
        tampered = copy.deepcopy(splits)
        row = tampered['train'][0]
        req = json.loads(row['request_json'])
        req['hour'] = (req['hour'] + 1) % 24
        row['request_json'] = json.dumps(req)
        with self.assertRaises(ValueError):
            validate(tampered)

        # write to a temporary location and read back to confirm load_rows works
        with tempfile.TemporaryDirectory() as directory:
            tmp_dir = Path(directory)
            for sp in ("train", "validation", "test"):
                write_csv(tmp_dir / f"{sp}.csv", splits[sp])
                rows = load_rows(tmp_dir / f"{sp}.csv")
                self.assertEqual(len(rows), len(splits[sp]))


if __name__ == "__main__":
    unittest.main()
