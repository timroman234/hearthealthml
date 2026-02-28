#!/bin/bash
# scripts/teardown_aws.sh
# Remove all AWS resources created by HeartHealthML
#
# WARNING: This will delete all resources including data!
#
# Usage:
#   ./scripts/teardown_aws.sh

set -e

# Configuration
BUCKET_NAME="${HEARTHEALTHML_BUCKET:-hearthealthml-artifacts}"
ECR_REPO_NAME="hearthealthml"
FUNCTION_NAME="hearthealthml-predict"
KEY_NAME="hearthealthml-key"
SECURITY_GROUP="hearthealthml-sg"
INSTANCE_NAME="hearthealthml-api"
REGION="${AWS_REGION:-us-east-1}"

echo "=========================================="
echo "HeartHealthML AWS Teardown"
echo "=========================================="
echo ""
echo "WARNING: This will delete ALL resources!"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5
echo ""

# Terminate EC2 instances
echo "Terminating EC2 instances..."
INSTANCE_IDS=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=${INSTANCE_NAME}" "Name=instance-state-name,Values=running,pending,stopped" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

if [ -n "$INSTANCE_IDS" ]; then
    aws ec2 terminate-instances --instance-ids $INSTANCE_IDS --region "$REGION"
    echo "Terminated: $INSTANCE_IDS"
    echo "Waiting for termination..."
    aws ec2 wait instance-terminated --instance-ids $INSTANCE_IDS --region "$REGION"
else
    echo "No running instances found"
fi

# Delete security group
echo ""
echo "Deleting security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SECURITY_GROUP}" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "None")

if [ "$SG_ID" != "None" ] && [ -n "$SG_ID" ]; then
    aws ec2 delete-security-group --group-id "$SG_ID" --region "$REGION" 2>/dev/null || echo "Could not delete security group (may be in use)"
    echo "Deleted security group: $SG_ID"
else
    echo "Security group not found"
fi

# Delete key pair
echo ""
echo "Deleting key pair..."
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" 2>/dev/null; then
    aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION"
    rm -f "${KEY_NAME}.pem"
    echo "Deleted key pair: $KEY_NAME"
else
    echo "Key pair not found"
fi

# Delete Lambda function
echo ""
echo "Deleting Lambda function..."
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null; then
    aws lambda delete-function --function-name "$FUNCTION_NAME" --region "$REGION"
    echo "Deleted Lambda: $FUNCTION_NAME"
else
    echo "Lambda function not found"
fi

# Delete ECR repository
echo ""
echo "Deleting ECR repository..."
if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null; then
    aws ecr delete-repository --repository-name "$ECR_REPO_NAME" --force --region "$REGION"
    echo "Deleted ECR: $ECR_REPO_NAME"
else
    echo "ECR repository not found"
fi

# Delete S3 bucket (must be empty first)
echo ""
echo "Deleting S3 bucket..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "Emptying bucket..."
    aws s3 rm "s3://${BUCKET_NAME}" --recursive
    aws s3api delete-bucket --bucket "$BUCKET_NAME" --region "$REGION"
    echo "Deleted bucket: $BUCKET_NAME"
else
    echo "S3 bucket not found"
fi

echo ""
echo "=========================================="
echo "Teardown Complete!"
echo "=========================================="
echo ""
echo "Note: IAM roles were not deleted. Remove manually if needed:"
echo "  aws iam delete-role --role-name hearthealthml-lambda-role"
echo "  aws iam delete-role --role-name hearthealthml-ec2-role"
