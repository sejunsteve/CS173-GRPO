"""Behavioral oracle examples written independently of generated predicates."""
import copy
import json
import unittest
from policy import evaluate, validate_request
from rewards import parse_output, score, access_reward

BASE = dict(role='editor', user_dept='ENG', resource_dept='ENG', clearance=3,
            classification=1, resource_state='active', action='read', is_owner=False,
            is_delegate=False, mfa=True, network='corporate', hour=12,
            emergency=False, purpose='routine')


class PolicyTests(unittest.TestCase):
    def check(self, rule, **changes):
        actual = evaluate(dict(BASE, **changes))
        allow = rule in ('R06', 'R08', 'R10', 'R11', 'R12')
        self.assertEqual(actual, {'decision': 'ALLOW' if allow else 'DENY', 'rule_id': rule})

    def test_all_rules_and_priority(self):
        cases = [
            ('R01', dict(resource_state='locked', clearance=0, role='admin', emergency=True, purpose='incident')),
            ('R02', dict(clearance=0, role='admin', emergency=True, purpose='incident')),
            ('R03', dict(resource_state='archived', action='edit', role='admin', emergency=True, purpose='incident')),
            ('R04', dict(network='public', classification=2, role='admin', emergency=True, purpose='incident')),
            ('R05', dict(action='edit', mfa=False, role='admin', emergency=True, purpose='incident')),
            ('R06', dict(role='admin', emergency=True, purpose='incident', action='edit', hour=2)),
            ('R07', dict(role='admin', action='delete', hour=2)),
            ('R08', dict(role='admin', user_dept='FIN')),
            ('R09', dict(user_dept='FIN')),
            ('R10', dict(role='guest', is_delegate=True)),
            ('R11', dict(role='viewer', action='edit', is_owner=True)),
            ('R12', dict(action='delete', is_owner=True)),
            ('R13', dict(role='guest')),
        ]
        for rule, changes in cases:
            with self.subTest(rule=rule):
                self.check(rule, **changes)

    def test_hour_boundaries(self):
        for hour, rule in [(8, 'R07'), (9, 'R11'), (16, 'R11'), (17, 'R07')]:
            with self.subTest(hour=hour):
                self.check(rule, action='edit', hour=hour)

    def test_delegate_and_owner_are_not_universal_permissions(self):
        self.check('R13', role='guest', action='edit', is_delegate=True)
        self.check('R13', role='viewer', action='delete', is_owner=True)
        self.check('R13', role='guest', is_owner=True)

    def test_clearance_equality_and_public_boundary(self):
        self.check('R10', clearance=1)
        self.check('R10', network='public', classification=1)
        self.check('R04', network='public', classification=2)

    def test_strict_input_types(self):
        for changes in [dict(hour=True), dict(clearance=False), dict(mfa='false'),
                        dict(hour=24), dict(classification=-1), dict(role='root'), dict(hour=12.0)]:
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                validate_request(dict(BASE, **changes))
        with self.assertRaises((ValueError, TypeError)):
            validate_request(dict(BASE, extra='unused'))


class RewardTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(score('{"decision":"ALLOW","rule_id":"R10"}', BASE), 1.0)
        self.assertEqual(score('{"decision":"ALLOW","rule_id":"R11"}', BASE), 0.4)
        self.assertEqual(score('{"decision":"DENY","rule_id":"R10"}', BASE), 0.0)

    def test_reward_hacking_attempts(self):
        invalid = ['[]', 'null', '1', '', 'ALLOW,R10',
            '```json\n{"decision":"ALLOW","rule_id":"R10"}\n```',
            '{"decision":"ALLOW","rule_id":"R10"} Explanation',
            '{"decision":"ALLOW","rule_id":"R10","reason":"yes"}',
            '{"decision":"DENY","decision":"ALLOW","rule_id":"R10"}',
            '{"decision":"allow","rule_id":"R10"}',
            '{"decision":"ALLOW","rule_id":"R99"}',
            '{"decision":"ALLOW","rule_id":10}',
            '{"decision":["ALLOW"],"rule_id":"R10"}',
            '{"decision":"ALLOW"}']
        for text in invalid:
            with self.subTest(text=text):
                self.assertIsNone(parse_output(text))
                self.assertEqual(score(text, BASE), 0.0)

    def test_key_order_whitespace_and_batch(self):
        answer = ' { "rule_id": "R10", "decision": "ALLOW" } '
        completions = [answer, [{'role': 'assistant', 'content': answer}]]
        self.assertEqual(access_reward(completions, [json.dumps(BASE)] * 2, expected_output=['lie'] * 2), [1.0, 1.0])
        with self.assertRaises(ValueError):
            access_reward(completions, [json.dumps(BASE)])


if __name__ == '__main__':
    unittest.main()
