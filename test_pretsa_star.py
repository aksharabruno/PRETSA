import unittest
from unittest.mock import MagicMock
import uuid
from anytree import Node, RenderTree
from pretsa_star import Pretsa_star

class TestPretsaStar(unittest.TestCase):

    def setUp(self):
        # Mock event log
        self.eventLog = MagicMock()
        self.eventLog.iterrows.return_value = iter([
            (0, {'Case ID': '1', 'impact': 'High', 'Activity': 'A', 'Duration': 5}),
            (1, {'Case ID': '2', 'impact': 'Low', 'Activity': 'B', 'Duration': 3}),
            (2, {'Case ID': '3', 'impact': 'Medium', 'Activity': 'C', 'Duration': 4}),
        ])
        self.pretsa_star = Pretsa_star(self.eventLog)
    
    def testCheckHomogeneousNodes(self):
        # Create a tree with homogeneous nodes
        root = Node("root", cases=set(['1', '2', '3']))
        child1 = Node("child1", parent=root, cases=set(['1']))
        child2 = Node("child2", parent=root, cases=set(['2']))
        child3 = Node("child3", parent=root, cases=set(['3']))

        self.pretsa_star._Pretsa_star__caseSensitiveValues = {
            '1': 'High',
            '2': 'High',
            '3': 'High'
        }
        self.pretsa_star._checkHomogenousNodes(root)
        #with self.assertRaises(Exception) as context:   #need to change this test
        #    self.pretsa_star._checkHomogenousNodes(root)
        #self.assertTrue('Homogeneity attack detected! The tree contains homogeneous nodes.' in str(context.exception))
       
    def testReplayAttack(self):
        # Mock parts of runPretsa that are not relevant to nonce validation
        self.pretsa_star._addDifferentialPrivateNosieToEnsureTCloseness = MagicMock()
        self.pretsa_star._checkPrivacyLevel = MagicMock()
        self.pretsa_star._checkLDiversity = MagicMock(return_value=False)

        # Generate a nonce and run Pretsa
        nonce1 = self.pretsa_star._generate_nonce()
        self.pretsa_star._validate_nonce(nonce1)

        # Try to reuse the same nonce
        with self.assertRaises(Exception) as context:
            self.pretsa_star._validate_nonce(nonce1)
        self.assertTrue('Replay attack detected! Nonce has already been used.' in str(context.exception))

        # Generate a new nonce and run Pretsa
        nonce2 = str(uuid.uuid4())
        self.pretsa_star._validate_nonce(nonce2)
    
if __name__ == '__main__':
    unittest.main()