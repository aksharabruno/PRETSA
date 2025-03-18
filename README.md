# PRETSA-Algorithm Family

This document details the extensions made to the Pretsa_star algorithm [^1] to enhance privacy and security through l-diversity implementation, privacy level assessment, and replay attack prevention.

Original repository: https://github.com/samadeusfp/PRETSA

## New Features

### L-Diversity Implementation

L-diversity extends k-anonymity by ensuring that sensitive values within each equivalence class (node) are sufficiently diverse [^2].

- `_extract_case_sensitive_values(self, eventLog)`:
  This function extracts sensitive values from the event log for each case ID. It iterates through the event log, maps each case ID to its "impact" value, and stores these mappings in a dictionary with case IDs as keys and impact values as values.

- `_checkHomogenousNodes(self, tree)`:
  This function identifies nodes with only one distinct sensitive value (homogeneous). It traverses the tree, looks at each node's cases, checks if all cases in a node have the same sensitive value, and calls the diversity improvement function when needed.

- `_modify_data_to_increase_diversity(self, node, l)`:
  This function increases diversity in homogeneous nodes to meet the l-diversity requirement. Initially, it determines current unique sensitive values in the node. If diversity is insufficient, attempts to group similar values using generalization. A generalized value categorization is done and synthetic cases with different values are added to meet the diversity threshold.

- `_group_similar_values(self, values, l)`:
  This function groups similar sensitive values to support generalization. It calculates the optimal number of groups needed based on target diversity level and partitions the values into these groups.

### Replay Attack Prevention

The implementation introduces a nonce-based mechanism to prevent replay attacks:

- `_generate_nonce(self)`:
  Uses the UUID4 standard to generate a cryptographically strong random identifier for each algorithm execution.

- `_validate_nonce(self, nonce)`:
  It checks if the provided nonce exists in the set of previously used nonces. If found, raises an exception indicating a replay attack attempt. If unique, adds the nonce to the set of used nonces.

### Privacy Level Assessment

- `_checkPrivacyLevel(self, tree)`:
  This function is used to determine the overall privacy level of the anonymized tree. The minimum number of cases in any node is considered as the k-anonymity metric. Nodes with only one distinct sensitive value is considered as the l-diversity metric. The overall score is a cumulation of both these metrics.

> [!Note]
> All these functions are implemented in pretsa_star.py

## How to run PRETSA\*

Install prerequisites

```
pip install -r requirements.txt
```

To run the pretsa algorithm, run the following script. Values of k and t can be changed in runPretsa.py.

```
python3 runPretsa.py
```

In addition, test_pretsa_star.py contains tests to see if the functionalities work as expected. To run,

```
python3 -m unittest test_pretsa_star.py
```

[^1]: Stephan A. Fahrenkrog-Petersen, Han van der Aa, and Matthias Weidlich. "Optimal event log sanitization for privacy-preserving process mining." [Link] (https://www.sciencedirect.com/science/article/abs/pii/S0169023X23000356?via%3Dihub)

[^2]: Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). "ℓ-Diversity: Privacy Beyond k-Anonymity." _ACM Transactions on Knowledge Discovery from Data (TKDD)_. [Link](https://www.cs.rochester.edu/u/muthuv/ldiversity-TKDD.pdf)
