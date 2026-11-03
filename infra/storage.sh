#!/usr/bin/env bash
# Persistent storage that outlives the instances: one private S3 bucket and an EC2 instance role
# that can read/write only that bucket. Idempotent; launch.sh runs it every time.
#
#   s3://<bucket>/compiled/<model>/<variant>/   compiled NxDI models (NEFFs + sharded weights), reused across boxes
#   s3://<bucket>/vllm/<model>/<variant>/       vllm-neuron compiled artifacts
#   s3://<bucket>/profiles/<run>/               raw NEFF/NTFF traces (too big for git)
#
# Writes infra/.state/storage.env (BUCKET, PROFILE). Delete everything with:
#   aws s3 rb s3://<bucket> --force && aws iam remove-role-from-instance-profile ... (see bottom of file)
set -euo pipefail
REGION=${REGION:-us-east-1}
NAME=${NAME:-neuron-lab}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STATE=$ROOT/infra/.state; mkdir -p "$STATE"
AWS="aws --region $REGION"

ACCOUNT=$($AWS sts get-caller-identity --query Account --output text)
BUCKET=${BUCKET:-$NAME-$ACCOUNT-$REGION}
ROLE=$NAME-ec2

if ! $AWS s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = us-east-1 ]; then $AWS s3api create-bucket --bucket "$BUCKET" >/dev/null
  else $AWS s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
  echo "created s3://$BUCKET"
fi
$AWS s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
$AWS s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration '{"Rules":[
  {"ID":"abort-partial-uploads","Status":"Enabled","Filter":{},"AbortIncompleteMultipartUpload":{"DaysAfterInitiation":3}},
  {"ID":"expire-raw-profiles","Status":"Enabled","Filter":{"Prefix":"profiles/"},"Expiration":{"Days":90}}]}'

NEW=0
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --description "neuron-lab instances: S3 access to $BUCKET only" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
      "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  echo "created role $ROLE"; NEW=1
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name s3-bucket --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
  {\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::$BUCKET\"},
  {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:AbortMultipartUpload\"],
   \"Resource\":\"arn:aws:s3:::$BUCKET/*\"}]}"
if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  echo "created instance profile $ROLE"; NEW=1
fi
# New IAM objects take a few seconds before EC2 accepts them.
[ $NEW = 1 ] && sleep 15

printf 'BUCKET=%s\nPROFILE=%s\n' "$BUCKET" "$ROLE" > "$STATE/storage.env"
echo "storage: s3://$BUCKET  instance profile: $ROLE"
$AWS s3 ls "s3://$BUCKET/compiled/" --recursive 2>/dev/null | grep manifest.txt | sed 's|^|  cached: |' || true

# Full removal (nothing else depends on these):
#   aws s3 rb s3://$BUCKET --force
#   aws iam remove-role-from-instance-profile --instance-profile-name $ROLE --role-name $ROLE
#   aws iam delete-instance-profile --instance-profile-name $ROLE
#   aws iam delete-role-policy --role-name $ROLE --policy-name s3-bucket && aws iam delete-role --role-name $ROLE
