# AWS S3 Setup Guide for HeartHealthML

Step-by-step guide to configure AWS S3 as shared DVC remote storage for team collaboration.

---

## Prerequisites

- AWS account (free tier eligible)
- AWS CLI installed (`aws --version`)

---

## Step 1: Create AWS Account (Skip if you have one)

1. Go to https://aws.amazon.com/free
2. Click "Create a Free Account"
3. Follow the signup process (requires credit card, but free tier won't charge)
4. Verify your email and phone number

---

## Step 2: Create S3 Bucket

### Option A: AWS Console (Web UI)

1. Log in to AWS Console: https://console.aws.amazon.com
2. Search for "S3" in the search bar
3. Click "Create bucket"
4. Configure:
   - **Bucket name**: `hearthealthml-data` (must be globally unique, add random suffix if taken)
   - **Region**: `us-east-1` (or closest to you)
   - **Block Public Access**: Keep ALL checked (private bucket)
   - Leave other settings as default
5. Click "Create bucket"

### Option B: AWS CLI

```bash
# Create bucket (add random suffix if name is taken)
aws s3 mb s3://hearthealthml-data-<your-initials> --region us-east-1

# Verify bucket was created
aws s3 ls
```

---

## Step 3: Create IAM User for DVC Access

### 3.1 Create IAM Policy

1. Go to IAM Console: https://console.aws.amazon.com/iam
2. Click "Policies" → "Create policy"
3. Click "JSON" tab and paste:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::hearthealthml-data-<your-suffix>",
                "arn:aws:s3:::hearthealthml-data-<your-suffix>/*"
            ]
        }
    ]
}
```

4. Click "Next"
5. Name it: `HeartHealthML-DVC-Policy`
6. Click "Create policy"

### 3.2 Create IAM User

1. In IAM Console, click "Users" → "Create user"
2. **User name**: `hearthealthml-dvc`
3. Click "Next"
4. Select "Attach policies directly"
5. Search for and select `HeartHealthML-DVC-Policy`
6. Click "Next" → "Create user"

### 3.3 Create Access Keys

1. Click on the user `hearthealthml-dvc`
2. Go to "Security credentials" tab
3. Under "Access keys", click "Create access key"
4. Select "Command Line Interface (CLI)"
5. Check the confirmation box, click "Next"
6. Click "Create access key"
7. **IMPORTANT**: Download the CSV or copy both:
   - Access key ID
   - Secret access key

   (You won't be able to see the secret key again!)

---

## Step 4: Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# Enter when prompted:
# AWS Access Key ID: <paste your access key>
# AWS Secret Access Key: <paste your secret key>
# Default region name: us-east-1
# Default output format: json

# Verify configuration
aws sts get-caller-identity
```

---

## Step 5: Update Your .env File

Update `F:\AIML_Apps\hearthealthml\.env`:

```bash
# .env - Local development environment variables

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=hearthealthml

# AWS
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
AWS_DEFAULT_REGION=us-east-1

# S3 Bucket (update with your actual bucket name)
S3_BUCKET=hearthealthml-data-<your-suffix>

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_NAME=logistic_regression
MODEL_VERSION=latest
```

---

## Step 6: Configure DVC to Use S3

```bash
# Navigate to project
cd F:/AIML_Apps/hearthealthml

# Add S3 remote (replace with your bucket name)
uv run dvc remote add -d s3remote s3://hearthealthml-data-<your-suffix>/dvc

# Set S3 remote as default (replaces localremote)
uv run dvc remote default s3remote

# Verify configuration
uv run dvc remote list
```

Expected output:
```
localremote    C:/dvc-storage
s3remote       s3://hearthealthml-data-<your-suffix>/dvc (default)
```

---

## Step 7: Push Data to S3

```bash
# Push existing tracked data to S3
uv run dvc push

# Verify data is in S3
aws s3 ls s3://hearthealthml-data-<your-suffix>/dvc/ --recursive
```

---

## Step 8: Commit DVC Config Changes

```bash
# Stage the updated DVC config
git add .dvc/config

# Commit
git commit -m "feat: configure S3 as DVC remote storage"

# Push to feature branch
git push
```

---

## Teammate Setup

Share these instructions with your teammate:

### For Your Teammate

1. **Get AWS credentials** (securely shared by team lead, NOT via Git)

2. **Configure AWS CLI**:
   ```bash
   aws configure
   # Enter the shared Access Key ID and Secret Access Key
   ```

3. **Clone the repo**:
   ```bash
   git clone <repo-url>
   cd hearthealthml
   ```

4. **Install dependencies**:
   ```bash
   uv sync --all-extras
   ```

5. **Pull data from S3**:
   ```bash
   uv run dvc pull
   ```

6. **Create local .env file** with shared credentials

---

## Security Best Practices

1. **Never commit credentials** - `.env` is in `.gitignore`
2. **Use IAM users** - Don't use root account credentials
3. **Minimal permissions** - Policy only allows S3 access to specific bucket
4. **Rotate keys periodically** - Create new keys and delete old ones
5. **Share credentials securely** - Use password manager or encrypted channel, NOT email/Slack

---

## Troubleshooting

### "Access Denied" error
- Verify IAM policy has correct bucket name
- Check AWS credentials are configured: `aws sts get-caller-identity`
- Ensure bucket exists: `aws s3 ls`

### "Bucket does not exist" error
- Check bucket name spelling in DVC remote config
- Verify bucket region matches your AWS config

### DVC push/pull hangs
- Check internet connection
- Verify S3 bucket is accessible: `aws s3 ls s3://your-bucket-name/`

---

## Cost Monitoring

AWS Free Tier includes:
- 5GB S3 storage
- 20,000 GET requests
- 2,000 PUT requests

Monitor usage at: https://console.aws.amazon.com/billing/home

---

## Quick Reference Commands

```bash
# Check DVC remote config
uv run dvc remote list

# Push data to S3
uv run dvc push

# Pull data from S3
uv run dvc pull

# Check what would be pushed/pulled
uv run dvc status

# List S3 bucket contents
aws s3 ls s3://your-bucket-name/ --recursive
```
