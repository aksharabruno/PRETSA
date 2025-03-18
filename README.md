# PRETSA-Algorithm Family
This document details the extensions made to the Pretsa_star algorithm to enhance privacy and security through l-diversity implementation, privacy level assessment, and replay attack prevention.

## New Features
### L-Diversity Implementation
L-diversity extends k-anonymity by ensuring that sensitive values within each equivalence class (node) are sufficiently diverse. This implementation adds the following capabilities:
- Homogeneous Node Detection
The system automatically detects nodes that contain only one distinct sensitive value (homogeneous nodes):
      --code
- Diversity Enhancement
When a homogeneous node is detected, the system employs a multi-strategy approach to increase diversity:
      --code

### Privacy Level Assessment
The extension includes a comprehensive privacy level assessment mechanism that evaluates both k-anonymity and l-diversity metrics:

This function:

Calculates the minimum number of cases per node (k-anonymity level)
Counts the number of homogeneous nodes (l-diversity violations)
Returns an overall privacy score that balances both metrics

### Replay Attack Prevention
The implementation introduces a nonce-based mechanism to prevent replay attacks:


# How to run PRETSA*
