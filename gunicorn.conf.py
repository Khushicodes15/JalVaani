"""
Gunicorn configuration for JalVaani AI API.

Key design for 100k concurrent users:
  - UvicornWorker: each worker runs a full async event loop.
    I/O waits (network, disk) never block other requests within a worker.
  - preload_app=True: the master process loads the 2.5 GB ML model once.
    Workers are forked from the master. Linux copy-on-write (COW) means
    workers share the model's memory pages as long as they only READ.
    Net effect: 4 workers cost ~1× model memory instead of 4×.
  - asyncio.to_thread inside each endpoint: CPU-bound ML inference runs
    in the worker's ThreadPoolExecutor (default: min(32, cpu+4) threads),
    keeping the event loop free to accept new connections while inference runs.

Scaling beyond one machine:
  - Run this behind Nginx (see nginx.conf) as a reverse proxy.
  - Spin up N identical containers/VMs, each running gunicorn.
  - Point a load balancer at them.
  - Switch cache.py to Redis for shared caching across instances.

Run:
  cd jalvaani_api
  gunicorn -c ../gunicorn.conf.py main:app
"""
import multiprocessing

# ── Bind ─────────────────────────────────────────────────────────────────────
bind = "0.0.0.0:8000"

# ── Workers ───────────────────────────────────────────────────────────────────
# async IO → fewer workers needed vs. sync. 2×CPU+1 is the standard rule.
worker_class = "uvicorn.workers.UvicornWorker"
workers      = multiprocessing.cpu_count() * 2 + 1

# Load app (and all 2.5 GB of ML models) in master BEFORE forking workers.
# Workers inherit via fork → COW → minimal extra memory per worker.
preload_app  = True

# ── Timeouts ──────────────────────────────────────────────────────────────────
timeout          = 120   # Generous: cold ML inference can be slow
keepalive        = 5     # HTTP keep-alive (seconds)
graceful_timeout = 30    # Wait this long for in-flight requests on SIGTERM

# ── Connection queue ──────────────────────────────────────────────────────────
backlog = 2048   # OS listen() queue length — increase if seeing "connection refused"

# ── Worker lifecycle ──────────────────────────────────────────────────────────
# Recycle workers to prevent slow memory drift (optional — disable if not needed)
max_requests        = 2000
max_requests_jitter = 200   # Stagger recycling so not all workers restart at once

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog      = "-"      # stdout
errorlog       = "-"      # stderr
loglevel       = "info"
capture_output = True
