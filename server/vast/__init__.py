"""Vast.ai GPU pool: rent GPU boxes on demand, reap them when idle.

The control plane never keeps an idle GPU running. `server.vast.pool` is a small
autoscaler that watches how many training workflows need a GPU and rents/reaps
ephemeral Vast.ai boxes to match, keeping recently-used boxes warm for reuse.

- `client`   — thin async wrapper over the Vast REST API.
- `registry` — Redis view of the boxes we manage (heartbeat + in-flight count),
               written by each GPU worker and read by the pool.
- `pool`     — the reconcile loop (`python -m server.vast.pool`).
"""
