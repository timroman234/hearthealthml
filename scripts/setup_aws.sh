#!/bin/bash
# scripts/setup_aws.sh
# Initial AWS infrastructure setup for HeartHealthML
#
# This script creates:
#   - S3 bucket for model artifacts
#   - ECR repository for Docker images
#   - IAM role for EC2 (optional)
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#
# Usage:
#   ./scripts/setup_aws.sh

set -e

# Configuration
BUCKET_NAME="${HEARTHEALTHML_BUCKET:-hearthealthml-artifacts}"
ECR_REPO_NAME="hearthealthml"
REGION="${AWS_REGION:-us-east-1}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=========================================="
echo "HeartHealthML AWS Setup"
echo "=========================================="
echo "Region: ${REGION}"
echo "Account: ${ACCOUNT_ID}"
echo "Bucket: ${BUCKET_NAME}"
echo ""

# Create S3 bucket for artifacts
echo "Setting up S3 bucket..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "Bucket '${BUCKET_NAME}' already exists"
else
    # Create bucket (different command for us-east-1)
    if [ "$REGION" = "us-east-1" ]; then
        aws s3api create-bucket \
            --bucket "$BUCKET_NAME" \
            --region "$REGION"
    else
        aws s3api create-bucket \
            --bucket "$BUCKET_NAME" \
            --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi
    echo "Bucket created: ${BUCKET_NAME}"
fi

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled
echo "Versioning enabled on bucket"

# Create ECR repository
echo ""
echo "Setting up ECR repository..."
if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null; then
    echo "ECR repository '${ECR_REPO_NAME}' already exists"
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$REGION" \
        --image-scanning-configuration scanOnPush=true
    echo "ECR repository created: ${ECR_REPO_NAME}"
fi

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "S3 Bucket: s3://${BUCKET_NAME}"
echo "ECR Repository: ${ECR_URI}"
echo ""
echo "Next steps:"
echo ""
echo "1. Upload model artifacts to S3:"
echo "   aws s3 sync models/ s3://${BUCKET_NAME}/models/"
echo "   aws s3 sync reports/ s3://${BUCKET_NAME}/reports/"
echo ""
echo "2. Build and push Docker image:"
echo "   aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_URI}"
echo "   docker build -t hearthealthml -f docker/Dockerfile.serve ."
echo "   docker tag hearthealthml:latest ${ECR_URI}:latest"
echo "   docker push ${ECR_URI}:latest"
echo ""
echo "3. Deploy to EC2:"
echo "   ./scripts/deploy_ec2.sh"
