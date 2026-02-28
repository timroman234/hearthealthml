#!/bin/bash
# scripts/push_to_ecr.sh
# Build and push Docker image to AWS ECR
#
# Usage:
#   ./scripts/push_to_ecr.sh [tag]
#
# Examples:
#   ./scripts/push_to_ecr.sh           # Push as 'latest'
#   ./scripts/push_to_ecr.sh v1.0.0    # Push as 'v1.0.0'

set -e

# Configuration
ECR_REPO_NAME="hearthealthml"
REGION="${AWS_REGION:-us-east-1}"
TAG="${1:-latest}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "=========================================="
echo "Push to ECR"
echo "=========================================="
echo "Repository: ${ECR_URI}"
echo "Tag: ${TAG}"
echo ""

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Build image
echo ""
echo "Building Docker image..."
docker build -t hearthealthml -f docker/Dockerfile.serve .

# Tag image
echo ""
echo "Tagging image..."
docker tag hearthealthml:latest "${ECR_URI}:${TAG}"

# Also tag as latest if not already
if [ "$TAG" != "latest" ]; then
    docker tag hearthealthml:latest "${ECR_URI}:latest"
fi

# Push image
echo ""
echo "Pushing to ECR..."
docker push "${ECR_URI}:${TAG}"

if [ "$TAG" != "latest" ]; then
    docker push "${ECR_URI}:latest"
fi

echo ""
echo "=========================================="
echo "Push Complete!"
echo "=========================================="
echo "Image: ${ECR_URI}:${TAG}"
echo ""
echo "To deploy to EC2:"
echo "  ./scripts/deploy_ec2.sh"
