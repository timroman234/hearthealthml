#!/bin/bash
# scripts/deploy_lambda.sh
# Deploy HeartHealthML Lambda function for serverless inference
#
# Prerequisites:
#   - AWS CLI configured
#   - Lambda execution role created
#   - Model artifacts uploaded to S3
#
# Usage:
#   ./scripts/deploy_lambda.sh

set -e

# Configuration
FUNCTION_NAME="hearthealthml-predict"
REGION="${AWS_REGION:-us-east-1}"
RUNTIME="python3.11"
HANDLER="handler.handler"
MEMORY_SIZE=512
TIMEOUT=30
BUCKET="${HEARTHEALTHML_BUCKET:-hearthealthml-artifacts}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/hearthealthml-lambda-role"

echo "=========================================="
echo "HeartHealthML Lambda Deployment"
echo "=========================================="
echo "Function: ${FUNCTION_NAME}"
echo "Region: ${REGION}"
echo "Memory: ${MEMORY_SIZE}MB"
echo ""

# Create deployment directory
DEPLOY_DIR=$(mktemp -d)
PACKAGE_DIR="${DEPLOY_DIR}/package"
mkdir -p "$PACKAGE_DIR"

echo "Building deployment package..."

# Install dependencies
pip install -q -t "$PACKAGE_DIR" \
    scikit-learn \
    pandas \
    numpy \
    joblib

# Copy handler
cp lambda/handler.py "$PACKAGE_DIR/"

# Create zip file
cd "$PACKAGE_DIR"
zip -q -r "${DEPLOY_DIR}/deployment.zip" .
cd - > /dev/null

DEPLOYMENT_ZIP="${DEPLOY_DIR}/deployment.zip"
ZIP_SIZE=$(du -h "$DEPLOYMENT_ZIP" | cut -f1)
echo "Deployment package size: ${ZIP_SIZE}"

# Check if function exists
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null; then
    echo "Updating existing function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://${DEPLOYMENT_ZIP}" \
        --region "$REGION" > /dev/null

    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --memory-size "$MEMORY_SIZE" \
        --timeout "$TIMEOUT" \
        --environment "Variables={MODEL_BUCKET=${BUCKET},MODEL_KEY=models/logistic_regression_v1.0.3/model.joblib,PREPROCESSOR_KEY=models/logistic_regression_v1.0.3/preprocessor.joblib}" \
        --region "$REGION" > /dev/null
else
    echo "Creating new function..."

    # Check if role exists
    if ! aws iam get-role --role-name hearthealthml-lambda-role 2>/dev/null; then
        echo "Creating Lambda execution role..."

        # Create role
        aws iam create-role \
            --role-name hearthealthml-lambda-role \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' > /dev/null

        # Attach basic execution policy
        aws iam attach-role-policy \
            --role-name hearthealthml-lambda-role \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

        # Attach S3 read policy
        aws iam attach-role-policy \
            --role-name hearthealthml-lambda-role \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

        # Wait for role to propagate
        echo "Waiting for IAM role to propagate..."
        sleep 10
    fi

    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --handler "$HANDLER" \
        --role "$ROLE_ARN" \
        --zip-file "fileb://${DEPLOYMENT_ZIP}" \
        --timeout "$TIMEOUT" \
        --memory-size "$MEMORY_SIZE" \
        --environment "Variables={MODEL_BUCKET=${BUCKET},MODEL_KEY=models/logistic_regression_v1.0.3/model.joblib,PREPROCESSOR_KEY=models/logistic_regression_v1.0.3/preprocessor.joblib}" \
        --region "$REGION" > /dev/null
fi

# Clean up
rm -rf "$DEPLOY_DIR"

# Get function ARN
FUNCTION_ARN=$(aws lambda get-function \
    --function-name "$FUNCTION_NAME" \
    --region "$REGION" \
    --query 'Configuration.FunctionArn' \
    --output text)

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Function ARN: ${FUNCTION_ARN}"
echo ""
echo "Test the function:"
echo "  aws lambda invoke --function-name ${FUNCTION_NAME} \\"
echo "    --payload '{\"age\":55,\"sex\":1,\"cp\":0,\"trestbps\":140,\"chol\":250,\"fbs\":0,\"restecg\":1,\"thalach\":150,\"exang\":0,\"oldpeak\":1.5,\"slope\":1,\"ca\":0,\"thal\":2}' \\"
echo "    --cli-binary-format raw-in-base64-out \\"
echo "    response.json && cat response.json"
echo ""
echo "To create API Gateway endpoint, see AWS Console or use SAM/Serverless Framework"
