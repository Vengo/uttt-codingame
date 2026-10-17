# Problems & Challenges: `uttt-tests/Cargo.toml`

This document details dependency and build constraints for `uttt-tests/Cargo.toml`.

---

## 1. Challenges
- Ensure lightweight test dependencies while enabling deep integration testing against the path-dependent `uttt` crate.
- Maintaining zero external dependencies beyond serde and rand for reproducible CI/local test runs.
