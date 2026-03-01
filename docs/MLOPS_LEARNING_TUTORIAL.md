# MLOps Learning Tutorial: End-to-End Model Change Walkthrough

> **A hands-on guide to the full MLOps lifecycle through a real model improvement**

---

## Part 1: Introduction

### What We're Building

This tutorial walks you through a **complete MLOps workflow** by making a single, meaningful change to our heart disease prediction model. You'll experience every stage of the pipeline:

```
Code Change -> Local Testing -> Git/PR -> CI/CD -> Docker -> Deploy -> Monitor
```

By the end, you'll understand not just *what* each step does, but *why* it matters.

### The Change We're Making

We're adding **class weight balancing** to our logistic regression model:

```python
# Adding: "class_weight": "balanced"
```

This one-line change addresses a real ML problem (class imbalance) and touches every part of our pipeline.

### Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.11+ with `uv` installed
- [ ] Docker Desktop running
- [ ] Git configured with GitHub access
- [ ] AWS CLI configured (for deployment sections)
- [ ] This repository cloned locally

### Pipeline Overview

```
                            MLOps Pipeline Flow
+----------------------------------------------------------------------------+
|                                                                            |
|   1. CODE              2. TEST              3. COMMIT           4. CI/CD   |
|   +--------+          +--------+           +--------+          +--------+  |
|   | Edit   |   --->   | pytest |   --->    | git    |   --->   | GitHub |  |
|   | train  |          | local  |           | push   |          | Actions|  |
|   | .py    |          | MLflow |           | PR     |          | checks |  |
|   +--------+          +--------+           +--------+          +--------+  |
|       |                   |                    |                    |      |
|       v                   v                    v                    v      |
|   5. DOCKER           6. PUSH              7. DEPLOY           8. MONITOR |
|   +--------+          +--------+           +--------+          +--------+  |
|   | Build  |   --->   | ECR    |   --->    | EC2    |   --->   | Grafana|  |
|   | image  |          | push   |           | deploy |          | Prom   |  |
|   | test   |          |        |           |        |          |        |  |
|   +--------+          +--------+           +--------+          +--------+  |
|                                                                            |
+----------------------------------------------------------------------------+
```

---

## Part 2: Understanding the Problem

### Class Imbalance in Medical ML

Heart disease datasets often have **imbalanced classes**:
- ~45% positive cases (heart disease present)
- ~55% negative cases (no heart disease)

While this seems mild, it can bias models toward the majority class.

### Why This Matters for Patients

In medical ML, **false negatives are dangerous**:
- Missing a heart disease diagnosis = patient doesn't get treatment
- False positive = extra tests (inconvenient but safer)

We need high **recall** (sensitivity) - catching most true positive cases.

### How `class_weight="balanced"` Solves It

Scikit-learn's `class_weight="balanced"` automatically:
1. Calculates inverse class frequency weights
2. Penalizes misclassification of minority class more heavily
3. Formula: `weight = n_samples / (n_classes * np.bincount(y))`

### Expected Impact

| Metric    | Before | After (Expected) |
|-----------|--------|------------------|
| Accuracy  | ~85%   | ~83% (slight drop)|
| Precision | ~86%   | ~82%             |
| **Recall**| ~82%   | **~88%** (improvement)|
| F1        | ~84%   | ~85%             |

Trade-off: Slightly more false positives, but fewer missed diagnoses.

---

## Part 3: Making the Code Change

### Step 3.1: Create Feature Branch

```bash
# Ensure you're on latest main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feat/add-class-weight-balancing
```

**WHY:** Feature branches isolate changes, enable code review, and provide easy rollback if something breaks.

### Step 3.2: Make the Code Change

**File:** `src/models/train.py`
**Lines:** 22-30

Open the file and locate the `DEFAULT_PARAMS` dictionary:

```python
# BEFORE (lines 22-30):
DEFAULT_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 200,
        "random_state": 42,
    },
}
```

Add the `class_weight` parameter:

```python
# AFTER (lines 22-31):
DEFAULT_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 200,
        "random_state": 42,
        "class_weight": "balanced",  # Handle imbalanced heart disease data
    },
}
```

### Step 3.3: Understand What Changed

The diff is minimal:

```diff
 DEFAULT_PARAMS = {
     "logistic_regression": {
         "C": 1.0,
         "penalty": "l2",
         "solver": "liblinear",
         "max_iter": 200,
         "random_state": 42,
+        "class_weight": "balanced",  # Handle imbalanced heart disease data
     },
 }
```

**How scikit-learn uses this:**
- During `model.fit()`, samples are weighted inversely to class frequency
- Minority class samples have higher weight in the loss function
- Model learns to "care more" about getting minority class right

---

## Part 4: Local Validation

### Step 4.1: Run Pre-commit Hooks

```bash
uv run pre-commit run --all-files
```

**Expected output:**
```
black....................................................................Passed
ruff.....................................................................Passed
check yaml...............................................................Passed
```

**WHY:** Pre-commit hooks enforce code style and catch obvious issues before they enter version control. Fixing formatting issues *after* code review wastes everyone's time.

### Step 4.2: Run Unit Tests

```bash
uv run pytest tests/ -v
```

**Expected output:**
```
tests/test_data.py::test_load_raw_data PASSED
tests/test_data.py::test_validate_schema PASSED
tests/test_features.py::test_engineer_features PASSED
tests/test_models.py::test_get_model PASSED
tests/test_models.py::test_train_model PASSED
...
========================= X passed in Y.XXs =========================
```

**WHY:** Tests verify our change doesn't break existing functionality. The `test_get_model` test specifically checks that model creation works with default parameters.

### Step 4.3: Establish Baseline Metrics

Before running with our change, record baseline metrics:

```bash
# Run pipeline WITHOUT our change (revert temporarily)
git stash
uv run python main.py

# Note the metrics, e.g.:
# Test Accuracy: 0.8478
# Test ROC-AUC: 0.9152
# Recall: 0.8205
```

**WHY:** We need a baseline to measure the impact of our change.

### Step 4.4: Run Pipeline with Change

```bash
# Restore our change
git stash pop

# Run pipeline WITH our change
uv run python main.py
```

**Expected output:**
```
=====================================
Starting HeartHealthML Pipeline
=====================================
Stage 1: Loading data
Stage 2: Validating data
Stage 3: Engineering features
Stage 4: Splitting data
Stage 5: Preprocessing features
Stage 6: Training model
  Created logistic_regression with params: {..., 'class_weight': 'balanced'}
Stage 7: Finding optimal threshold
Stage 8: Evaluating model
Stage 9: Registering model
=====================================
Pipeline completed successfully!
Model: logistic_regression v2
Test Accuracy: 0.8261
Test ROC-AUC: 0.9145
=====================================
```

**Compare results:**
| Metric   | Before  | After   | Change   |
|----------|---------|---------|----------|
| Accuracy | 0.8478  | 0.8261  | -2.6%    |
| ROC-AUC  | 0.9152  | 0.9145  | -0.1%    |
| Recall   | 0.8205  | 0.8750  | **+6.6%**|

**WHY:** The recall improvement shows we're catching more true positive cases - exactly what we wanted for medical diagnosis.

### Step 4.5: View Results in MLflow

```bash
# Start MLflow UI
uv run mlflow ui
```

Open http://localhost:5000 in your browser.

**What to look for:**
1. Compare runs side-by-side
2. View logged parameters (confirm `class_weight` is logged)
3. Check confusion matrix artifacts
4. Verify model is registered

**WHY:** MLflow provides visual comparison and artifact tracking, making it easy to see the impact of changes over time.

---

## Part 5: Committing and Code Review

### Step 5.1: Stage and Commit

```bash
# Check what changed
git status
git diff src/models/train.py

# Stage the change
git add src/models/train.py

# Commit with descriptive message
git commit -m "feat: add class weight balancing for imbalanced data

- Add class_weight='balanced' to logistic regression defaults
- Improves recall from ~82% to ~88% for heart disease detection
- Trade-off: slight accuracy decrease (~2%) is acceptable for medical use case"
```

**WHY:** Atomic commits with clear messages make history easy to understand. The "why" in commit messages helps future maintainers (including future you).

### Step 5.2: Push and Create PR

```bash
# Push branch to remote
git push -u origin feat/add-class-weight-balancing

# Create pull request
gh pr create \
  --title "feat: add class weight balancing for imbalanced data" \
  --body "## Summary
- Add \`class_weight='balanced'\` to logistic regression model
- Addresses class imbalance in heart disease dataset

## Impact
| Metric   | Before | After  |
|----------|--------|--------|
| Accuracy | 84.78% | 82.61% |
| Recall   | 82.05% | 87.50% |
| ROC-AUC  | 91.52% | 91.45% |

## Why the trade-off is acceptable
For medical diagnosis, **high recall is critical** - we want to minimize false negatives (missed diagnoses). The slight accuracy decrease is acceptable.

## Test plan
- [x] Unit tests pass locally
- [x] Pipeline runs successfully
- [x] Metrics logged to MLflow
- [ ] CI/CD pipeline passes
- [ ] Docker build succeeds"
```

**WHY:** PRs enable code review, trigger CI/CD, and document changes with context.

### Step 5.3: Code Review Process

**What reviewers look for:**
1. **Correctness:** Does the change do what it claims?
2. **Side effects:** Could this break anything?
3. **Tests:** Are changes covered by tests?
4. **Documentation:** Is the "why" clear?

**Responding to feedback:**
```bash
# Make requested changes
git add .
git commit -m "address review feedback: add comment explaining class weights"
git push
```

**WHY:** Code review catches bugs, shares knowledge, and improves code quality.

---

## Part 6: CI/CD Pipeline

### Step 6.1: Watch CI Run

```bash
# List workflow runs
gh run list

# Watch the latest run
gh run watch
```

**WHY:** Automated CI catches issues before merge, ensuring main branch stays healthy.

### Step 6.2: Understanding CI Checks

Our CI pipeline (`.github/workflows/ci.yml`) runs:

```yaml
jobs:
  lint:
    # Black formatting check
    # Ruff linting

  test:
    # pytest with coverage
    # Minimum coverage threshold

  validate:
    # Data validation checks

  build:
    # Docker image builds
```

**Each stage provides value:**

| Stage    | What it catches                          |
|----------|------------------------------------------|
| Lint     | Style issues, unused imports, typos      |
| Test     | Logic bugs, regressions, edge cases      |
| Validate | Data schema issues, missing files        |
| Build    | Dependency issues, Docker problems       |

### Step 6.3: Handling CI Failures

If CI fails, diagnose with:

```bash
# View logs
gh run view --log-failed

# Common fixes:
# - Lint failures: uv run black . && uv run ruff check --fix .
# - Test failures: uv run pytest tests/ -v --tb=short
# - Build failures: docker build -f docker/Dockerfile.serve .
```

**WHY:** CI failures are learning opportunities. They catch issues before they reach production.

### Step 6.4: Merge to Main

After approval and CI passes:

```bash
# Squash merge to keep history clean
gh pr merge --squash

# Clean up local branch
git checkout main
git pull
git branch -d feat/add-class-weight-balancing
```

**WHY:** Squash merging keeps main branch history clean while preserving full context in the PR.

---

## Part 7: Docker Build & Test

### Step 7.1: Build New Docker Image

```bash
# Build the serving image
docker build -t hearthealthml-serve:latest -f docker/Dockerfile.serve .
```

**Expected output:**
```
[+] Building 45.2s (12/12) FINISHED
 => [builder 1/3] FROM python:3.11-slim
 => [builder 2/3] COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
 => [builder 3/3] RUN uv pip install --system scikit-learn pandas...
 => [runtime 1/4] FROM python:3.11-slim
 => [runtime 2/4] COPY --from=builder /usr/local/lib/python3.11/site-packages...
 => [runtime 3/4] COPY src/ ./src/
 => [runtime 4/4] COPY api/ ./api/
 => exporting to image
```

**WHY:** Containerization ensures the model runs identically everywhere - your laptop, CI, staging, production.

### Step 7.2: Test Locally with Docker Compose

```bash
# Start all services
docker compose -f docker/docker-compose.yml up -d

# Check containers are running
docker compose -f docker/docker-compose.yml ps

# Expected output:
# NAME                     STATUS
# hearthealthml-api        Up (healthy)
# hearthealthml-mlflow     Up (healthy)
```

Test the API:

```bash
# Health check
curl http://localhost:8000/health

# Expected: {"status":"healthy","model_loaded":true,"version":"2"...}

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }'

# Expected: {"prediction":1,"probability":0.72,"risk_level":"high"...}
```

**WHY:** Testing containers locally catches deployment issues before they reach production.

### Step 7.3: Push to Container Registry (ECR)

```bash
# Run the push script
./scripts/push_to_ecr.sh

# Or with a specific tag
./scripts/push_to_ecr.sh v1.1.0
```

**What the script does:**
1. Authenticates with AWS ECR
2. Builds the Docker image
3. Tags with ECR repository URL
4. Pushes to AWS

**Expected output:**
```
==========================================
Push to ECR
==========================================
Repository: 123456789.dkr.ecr.us-east-1.amazonaws.com/hearthealthml
Tag: latest

Logging in to ECR...
Building Docker image...
Tagging image...
Pushing to ECR...
==========================================
Push Complete!
==========================================
```

**WHY:** Container registries store versioned images that can be deployed anywhere.

---

## Part 8: Deployment

### Step 8.1: Deploy to EC2 (Staging)

```bash
# Deploy using the script
./scripts/deploy_ec2.sh
```

**What the script does:**
1. Creates EC2 key pair (if needed)
2. Creates security group (ports 22, 8000)
3. Launches t2.micro instance (free tier)
4. Runs user-data script to pull and run container

**Expected output:**
```
==========================================
HeartHealthML EC2 Deployment
==========================================
Region: us-east-1
Account: 123456789012
Instance Type: t2.micro

Creating key pair...
Creating security group...
Launching EC2 instance...
==========================================
Deployment Complete!
==========================================
Instance ID: i-0abc123def456
Public IP:   54.123.45.67

SSH Access:
  ssh -i hearthealthml-key.pem ec2-user@54.123.45.67

API Endpoints (wait ~2 min for startup):
  Health: http://54.123.45.67:8000/health
  Docs:   http://54.123.45.67:8000/docs
  API:    http://54.123.45.67:8000/predict
```

**WHY:** Staging environments validate changes in a production-like setting before going live.

### Step 8.2: Run Health Checks

```bash
# Wait for instance to initialize (~2 minutes)
sleep 120

# Health check
curl http://<EC2_PUBLIC_IP>:8000/health

# Test prediction on deployed model
curl -X POST http://<EC2_PUBLIC_IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":140,"chol":250,"fbs":0,"restecg":0,"thalach":150,"exang":0,"oldpeak":1.5,"slope":1,"ca":0,"thal":2}'
```

**WHY:** Post-deployment health checks verify the service is actually running and responding correctly.

### Step 8.3: Monitor with Prometheus/Grafana

Start monitoring stack:

```bash
docker compose -f docker/docker-compose.yml --profile monitoring up -d
```

Access dashboards:
- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090

**Key metrics to watch:**
- `hearthealthml_predictions_total` - prediction volume
- `hearthealthml_prediction_latency_seconds` - response times
- `hearthealthml_requests_total` - HTTP request counts

**WHY:** Monitoring catches performance degradation and errors that tests might miss.

### Step 8.4: Production Deployment Considerations

For actual production:

1. **Approval gates:** Require manual approval before prod deploy
2. **Blue-green deployment:** Run old and new versions simultaneously
3. **Canary releases:** Route small % of traffic to new version first
4. **Rollback plan:** Know how to quickly revert if issues arise

```bash
# Rollback command (if needed)
docker pull ${ECR_REPO}:previous-version
docker stop hearthealthml-api
docker run -d --name hearthealthml-api -p 8000:8000 ${ECR_REPO}:previous-version
```

**WHY:** Production needs extra safeguards because downtime and bugs affect real users.

---

## Part 9: Summary & Key Takeaways

### Complete Workflow Diagram

```
+------------------+     +------------------+     +------------------+
|   DEVELOPMENT    |     |   INTEGRATION    |     |   DEPLOYMENT     |
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
| 1. Create branch | --> | 5. Push branch   | --> | 8. Build Docker  |
|                  |     |                  |     |                  |
| 2. Edit code     | --> | 6. Create PR     | --> | 9. Push to ECR   |
|                  |     |                  |     |                  |
| 3. Run tests     | --> | 7. CI checks     | --> | 10. Deploy EC2   |
|                  |     |                  |     |                  |
| 4. Run pipeline  | --> | 7b. Code review  | --> | 11. Health check |
|                  |     |                  |     |                  |
|                  |     | 7c. Merge        | --> | 12. Monitor      |
+------------------+     +------------------+     +------------------+
```

### Key Commands Cheat Sheet

```bash
# Development
git checkout -b feat/my-feature     # Create feature branch
uv run pre-commit run --all-files   # Check code style
uv run pytest tests/ -v             # Run tests
uv run python main.py               # Run ML pipeline
uv run mlflow ui                    # View MLflow dashboard

# Git/GitHub
git add <file>                      # Stage changes
git commit -m "message"             # Commit
git push -u origin <branch>         # Push branch
gh pr create                        # Create pull request
gh run watch                        # Watch CI pipeline
gh pr merge --squash                # Merge PR

# Docker
docker build -t name -f Dockerfile .  # Build image
docker compose up -d                   # Start services
docker compose ps                      # Check status
./scripts/push_to_ecr.sh              # Push to AWS ECR

# Deploy
./scripts/deploy_ec2.sh             # Deploy to EC2
curl http://<ip>:8000/health        # Health check
```

### What We Learned at Each Stage

| Stage           | Key Learning                                             |
|-----------------|----------------------------------------------------------|
| Code Change     | Small changes can have significant ML impact             |
| Pre-commit      | Automate style enforcement to focus on logic             |
| Unit Tests      | Tests protect against regressions                        |
| ML Pipeline     | Always compare before/after metrics                      |
| MLflow          | Experiment tracking enables reproducibility              |
| Git/PR          | Version control + review = quality code                  |
| CI/CD           | Automation catches human errors                          |
| Docker          | Containers ensure consistent environments                |
| ECR/Deploy      | Infrastructure as code enables repeatable deploys        |
| Monitoring      | Observability catches issues tests can't                 |

### Critical Files Reference

| File                          | Purpose                            |
|-------------------------------|------------------------------------|
| `src/models/train.py`         | Model training logic (we edited)   |
| `configs/config.yaml`         | Pipeline configuration             |
| `main.py`                     | Pipeline entry point               |
| `tests/test_models.py`        | Model unit tests                   |
| `docker/Dockerfile.serve`     | API container definition           |
| `docker/docker-compose.yml`   | Multi-container orchestration      |
| `scripts/push_to_ecr.sh`      | ECR push automation                |
| `scripts/deploy_ec2.sh`       | EC2 deployment automation          |

---

## Next Steps

Now that you've completed one full cycle, try:

1. **Add a test** for the class weight parameter in `tests/test_models.py`
2. **Enable hyperparameter tuning** with `uv run python main.py --tune`
3. **Add monitoring alerts** for prediction latency spikes
4. **Implement A/B testing** between balanced and unbalanced models

---

*Generated for HeartHealthML - A learning-focused ML project*
