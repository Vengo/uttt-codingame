# Problems & Challenges: `uttt/src/dfpn/mod.rs`

This document details implementation-level challenges in `uttt/src/dfpn/mod.rs`.

---

## 1. Challenges
- **Threshold Oscillation**: Classic DF-PN can suffer from deep thrashing if thresholds $(th_{pn}, th_{dn})$ are not expanded efficiently.
- **Node Budget Control**: Preventing search from blocking indefinitely on complex indeterminate endgames by enforcing `max_nodes` limits.
- **Child Proof Conversion**: Correctly translating child proof numbers (which are from defender's AND perspective) into attacker's effective proof numbers ($pn_{\text{eff}} = dn_{\text{child}}, dn_{\text{eff}} = pn_{\text{child}}$).
