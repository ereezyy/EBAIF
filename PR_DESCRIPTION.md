# 🔒 Security Fix: Predictable Consensus Engine IDs

## 🎯 What
Fixed predictable ID generation in `src/ebaif/consensus/engine.py`. The `engine_id` and `proposal_id` were generated using `time.time()`, making them predictable.

## ⚠️ Risk
Predictable IDs could allow attackers to spoof consensus engines or interfere with proposal validation by guessing IDs.

## 🛡️ Solution
Replaced `time.time()` based generation with `uuid.uuid4()`, which provides cryptographically secure random identifiers.

## verification
- [x] Verified `uuid` import added.
- [x] Verified `engine_id` uses `uuid.uuid4()`.
- [x] Verified `proposal_id` uses `uuid.uuid4()`.
- [x] Ran `py_compile` to ensure no syntax errors.
