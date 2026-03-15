# Technical Debt / TODO

Items to address in future cleanup sprints.

## Deprecation Warnings

### 1. `datetime.utcnow()` deprecation (54 warnings)

**Files:**
- `src/monitoring/model_monitor.py:71`
- `src/monitoring/model_monitor.py:154`
- `src/monitoring/prediction_logger.py:27`
- `tests/test_monitoring.py:108, 112`

**Fix:** Replace `datetime.utcnow()` with `datetime.now(datetime.UTC)`

```python
# Before
from datetime import datetime
datetime.utcnow()

# After
from datetime import datetime, UTC
datetime.now(UTC)
```

### 2. Scikit-learn `penalty` parameter (5 warnings)

**Files:**
- `src/models/train.py` (DEFAULT_PARAMS and PARAM_GRIDS)

**Warning:** `'penalty' was deprecated in version 1.8 and will be removed in 1.10`

**Fix:** Use `l1_ratio` instead of `penalty` when sklearn 1.10 is released. Monitor sklearn changelog.

## Ruff Configuration

### 3. Deprecated pyproject.toml settings

**File:** `pyproject.toml`

**Warning:** Top-level linter settings deprecated in favour of `lint` section.

**Fix:** Update ruff config:
```toml
# Before
[tool.ruff]
select = ["E", "F"]

# After
[tool.ruff.lint]
select = ["E", "F"]
```

## Notes

- Created: 2026-03-01
- These don't block development - all tests pass
- Address when doing maintenance or upgrading dependencies
