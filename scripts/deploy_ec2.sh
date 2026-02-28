#!/bin/bash
# scripts/deploy_ec2.sh
# Deploy HeartHealthML API to AWS EC2 (Free Tier eligible)
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker image pushed to ECR
#
# Usage:
#   ./scripts/deploy_ec2.sh

set -e

# Configuration
INSTANCE_TYPE="t2.micro"  # Free tier eligible
AMI_ID="ami-0c7217cdde317cfec"  # Amazon Linux 2023 (us-east-1)
KEY_NAME="hearthealthml-key"
SECURITY_GROUP="hearthealthml-sg"
INSTANCE_NAME="hearthealthml-api"
REGION="${AWS_REGION:-us-east-1}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/hearthealthml"

echo "=========================================="
echo "HeartHealthML EC2 Deployment"
echo "=========================================="
echo "Region: ${REGION}"
echo "Account: ${ACCOUNT_ID}"
echo "Instance Type: ${INSTANCE_TYPE}"
echo ""

# Check if key pair exists
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" 2>/dev/null; then
    echo "Key pair '${KEY_NAME}' already exists"
else
    echo "Creating key pair..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' \
        --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    echo "Key saved to ${KEY_NAME}.pem"
fi

# Check if security group exists
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SECURITY_GROUP}" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP" \
        --description "HeartHealthML API security group" \
        --region "$REGION" \
        --query 'GroupId' \
        --output text)

    # Allow SSH (port 22)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"

    # Allow API (port 8000)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"

    echo "Security group created: ${SG_ID}"
else
    echo "Security group '${SECURITY_GROUP}' already exists: ${SG_ID}"
fi

# Generate user data script with account-specific values
USER_DATA=$(cat <<EOF
#!/bin/bash
# HeartHealthML EC2 User Data Script

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install AWS CLI v2 (if not present)
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
fi

# Wait for Docker to be ready
sleep 5

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

# Pull and run container
docker pull ${ECR_REPO}:latest
docker run -d \
    --name hearthealthml-api \
    -p 8000:8000 \
    --restart unless-stopped \
    -e MODEL_NAME=logistic_regression \
    -e MODEL_VERSION=latest \
    ${ECR_REPO}:latest

# Log completion
echo "HeartHealthML API deployed successfully" | logger
EOF
)

# Launch EC2 instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$REGION" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
    --user-data "$USER_DATA" \
    --iam-instance-profile Name=hearthealthml-ec2-role \
    --query 'Instances[0].InstanceId' \
    --output text 2>/dev/null || \
    # Retry without IAM profile if it doesn't exist
    aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --region "$REGION" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
        --user-data "$USER_DATA" \
        --query 'Instances[0].InstanceId' \
        --output text)

echo "Instance ID: ${INSTANCE_ID}"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Instance ID: ${INSTANCE_ID}"
echo "Public IP:   ${PUBLIC_IP}"
echo ""
echo "SSH Access:"
echo "  ssh -i ${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo "API Endpoints (wait ~2 min for startup):"
echo "  Health: http://${PUBLIC_IP}:8000/health"
echo "  Docs:   http://${PUBLIC_IP}:8000/docs"
echo "  API:    http://${PUBLIC_IP}:8000/predict"
echo ""
echo "To terminate:"
echo "  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}"
