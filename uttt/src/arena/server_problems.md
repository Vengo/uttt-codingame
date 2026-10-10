# Problems & Challenges: `uttt/src/arena/server.rs`

This document details web server and REST API challenges in `uttt/src/arena/server.rs`.

---

## 1. Challenges
- **Zero Heavy Web Dependencies**: Implementing an HTTP/1.1 server using only `std::net::TcpListener` and `std::thread::spawn` without bulky frameworks (actix, axum, warp) to ensure fast compilation and zero binary bloat.
- **REST Request Parsing**: Safely parsing HTTP request methods, paths, and JSON request bodies without crashing on malformed payloads.
- **Thread Safety**: Safely sharing `legal_moves` lookup tables and `ZobristConstants` across concurrent request worker threads using `Arc`.
