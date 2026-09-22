from __future__ import annotations

import threading


# Paddle predictors share one GPU in the initial deployment. Serialize the
# general OCR pipeline and the check-digit recognition-only model so concurrent
# HTTP requests cannot make two different predictors use the GPU at once.
PADDLE_RUNTIME_LOCK = threading.RLock()
