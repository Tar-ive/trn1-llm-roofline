#!/usr/bin/env bash
# Launch a self-destructing Trainium box that clones this repo, runs bench/run.sh,
# pushes results back to GitHub, and terminates itself after $MINUTES.
#
#   aws login                      # or any valid AWS credentials
#   ./infra/launch.sh              # defaults below; override via env vars
#   MINUTES=90 REGION=us-west-2 ./infra/launch.sh
#   MODE=compile ./infra/launch.sh  # CPU-only box (no Trn quota needed): compile for trn1, upload to S3, exit
set -euo pipefail

REGION=${REGION:-us-east-1}
MODE=${MODE:-run}                # run = benchmark/profile on Trainium; compile = precompile on a CPU instance
case "$MODE" in
  run)     ITYPE=${ITYPE:-trn1.2xlarge}; MIN_KEY=LAUNCH_MINUTES ;;
  compile) ITYPE=${ITYPE:-m7i.4xlarge};  MIN_KEY=COMPILE_LAUNCH_MINUTES ;;   # 16 vCPU / 64 GiB, ~$0.81/hr
  *) echo "MODE must be run or compile"; exit 1 ;;
esac
MINUTES=${MINUTES:-$(sed -n "s/^$MIN_KEY=\([0-9]*\).*/\1/p" "$(dirname "$0")/../bench/config.env")}; MINUTES=${MINUTES:-60}   # default from bench/config.env
AUTORUN=${AUTORUN:-1}            # 1 = start bench/run.sh automatically at boot
ROOT_GB=${ROOT_GB:-200}
NAME=${NAME:-neuron-lab}
AMI_PARAM=${AMI_PARAM:-/aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id}
# Neuron 2.29+ dropped Trn1/Inf2 from NxD Inference and vllm-neuron, so pin those to the last
# Neuron 2.28 DLAMI (NxDI 0.8 + vllm-neuron 0.4). Override with AMI_ID=ami-... or AMI_NAME=...
# Compile mode targets trn1 too, and compiled artifacts only load on the exact same SDK, so it gets the same AMI.
case "$ITYPE:$MODE" in
  trn1*|inf2*|*:compile) AMI_NAME=${AMI_NAME:-"Deep Learning AMI Neuron PyTorch Inference vLLM 0.13 (Ubuntu 24.04) 20260227"} ;;
esac
REPO=${REPO:-$(gh api user -q .login)/neuron-llm-profiling}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STATE=$ROOT/infra/.state          # gitignored: keys, rendered user-data, instance id
mkdir -p "$STATE"
AWS="aws --region $REGION"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Checking AWS credentials"
$AWS sts get-caller-identity --query Arn --output text || { echo "Run 'aws login' first."; exit 1; }

say "Preflight: $ITYPE availability and vCPU quota in $REGION (./infra/regions.sh scans all regions)"
AZS=$($AWS ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters Name=instance-type,Values="$ITYPE" --query 'InstanceTypeOfferings[].Location' --output text)
[ -n "$AZS" ] || { echo "$ITYPE is not offered in $REGION. Run ./infra/regions.sh"; exit 1; }
case "$ITYPE" in    # per-family on-demand vCPU quotas
  trn*) OD_QUOTA=${OD_QUOTA:-L-2C3B7624}; FAMILY="trn*" ;;
  inf*) OD_QUOTA=${OD_QUOTA:-L-1945791B}; FAMILY="inf*" ;;
  *)    OD_QUOTA=${OD_QUOTA:-L-1216C47A}; FAMILY="a*,c*,d*,h*,i*,m*,r*,t*,z*" ;;   # Standard (A, C, D, H, I, M, R, T, Z)
esac
VCPUS=$($AWS ec2 describe-instance-types --instance-types "$ITYPE" --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
QUOTA=$($AWS service-quotas get-service-quota --service-code ec2 --quota-code "$OD_QUOTA" \
  --query 'Quota.Value' --output text | cut -d. -f1)
INUSE=$($AWS ec2 describe-instances \
  --filters "Name=instance-type,Values=$FAMILY" Name=instance-state-name,Values=pending,running,stopping,shutting-down \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,CpuOptions.CoreCount,CpuOptions.ThreadsPerCore]' \
  --output text)
# Standard-family wildcards (i*, t*) also match inf*/trn*, which have their own quotas: drop those.
case "$ITYPE" in trn*|inf*) ;; *) INUSE=$(echo "$INUSE" | awk '$2 !~ /^(trn|inf)/') ;; esac
USED=$(echo "$INUSE" | awk 'NF {s += $4 * $5} END {print s + 0}')
echo "AZs: $AZS   on-demand quota ($OD_QUOTA): $QUOTA vCPUs, in use: $USED, need: $VCPUS"
if [ $((QUOTA - USED)) -lt "$VCPUS" ] && [ "$QUOTA" -ge "$VCPUS" ]; then
  echo "Quota is held by running Trn instances. Terminate one (or wait for auto-termination), then re-run:"
  echo "$INUSE" | awk 'NF {print "  " $1, $2, $3}'
  echo "  aws --region $REGION ec2 terminate-instances --instance-ids <id>"
  exit 1
fi
if [ "$QUOTA" -lt "$VCPUS" ]; then
  echo "Quota too low. Request it (approval can take hours to days), then re-run:"
  echo "  aws service-quotas request-service-quota-increase --region $REGION --service-code ec2 --quota-code $OD_QUOTA --desired-value $VCPUS"
  exit 1
fi

say "GitHub repo $REPO"
if ! git -C "$ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "Commit something in $ROOT first."; exit 1
fi
# The box runs whatever is on GitHub, so local edits must be committed and pushed first.
if [ -n "$(git -C "$ROOT" status --porcelain -- . ':!results')" ]; then
  git -C "$ROOT" status --short -- . ':!results'
  echo "Uncommitted changes above would NOT reach the instance. Commit them (or stash) and re-run."; exit 1
fi
if gh repo view "$REPO" >/dev/null 2>&1; then
  git -C "$ROOT" remote get-url origin >/dev/null 2>&1 || git -C "$ROOT" remote add origin "https://github.com/$REPO.git"
  git -C "$ROOT" pull --rebase origin "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"   # pick up results pushed by earlier boxes
  git -C "$ROOT" push -u origin HEAD
else
  gh repo create "$REPO" --private --source "$ROOT" --remote origin --push
fi
BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)

say "Ephemeral write-scoped deploy key (removed by infra/teardown.sh)"
# Drop keys left behind by earlier launches of this MODE (e.g. a run whose teardown never happened).
for OLD in $(gh repo deploy-key list --repo "$REPO" --json id,title -q ".[] | select(.title | startswith(\"$NAME-$MODE-\")) | .id"); do
  gh repo deploy-key delete "$OLD" --repo "$REPO" && echo "  deleted stale deploy key $OLD"
done
KEY_TITLE="$NAME-$MODE-$(date +%s)"   # per mode, so a run launch never revokes a live compile box's key
rm -f "$STATE"/deploy_key "$STATE"/deploy_key.pub
ssh-keygen -q -t ed25519 -N "" -C "$KEY_TITLE" -f "$STATE/deploy_key"
gh repo deploy-key add "$STATE/deploy_key.pub" --repo "$REPO" --allow-write --title "$KEY_TITLE"

say "SSH login key pair"
[ -f "$STATE/login_key" ] || ssh-keygen -q -t ed25519 -N "" -C "$NAME-login" -f "$STATE/login_key"
$AWS ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1 ||
  $AWS ec2 import-key-pair --key-name "$NAME" --public-key-material "fileb://$STATE/login_key.pub" >/dev/null

say "Persistent storage (S3 bucket + instance role)"
REGION=$REGION NAME=$NAME "$ROOT/infra/storage.sh"
source "$STATE/storage.env"   # BUCKET, PROFILE

say "Lab network (own VPC; SSH from this machine's current IP only)"
REGION=$REGION ITYPE=$ITYPE NAME=$NAME "$ROOT/infra/network.sh"
source "$STATE/network.env"   # VPC, SG, SUBNETS

say "Resolving Neuron DLAMI"
if [ -n "${AMI_ID:-}" ]; then
  AMI=$AMI_ID
elif [ -n "${AMI_NAME:-}" ]; then
  AMI=$($AWS ec2 describe-images --owners amazon --filters "Name=name,Values=$AMI_NAME" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  [ "$AMI" != "None" ] || { echo "No AMI named '$AMI_NAME' in $REGION"; exit 1; }
else
  AMI=$($AWS ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)
fi
$AWS ec2 describe-images --image-ids "$AMI" --query 'Images[0].[Name,Description]' --output text
read -r ROOT_DEV SNAP_GB < <($AWS ec2 describe-images --image-ids "$AMI" \
  --query 'Images[0].[RootDeviceName,BlockDeviceMappings[0].Ebs.VolumeSize]' --output text)
[ "$SNAP_GB" -gt "$ROOT_GB" ] && ROOT_GB=$SNAP_GB
echo "AMI $AMI  root $ROOT_DEV ${ROOT_GB}GB"

say "Rendering user-data"
KEY_B64=$(base64 < "$STATE/deploy_key" | tr -d '\n')
sed -e "s|__REPO__|$REPO|g" -e "s|__BRANCH__|$BRANCH|g" -e "s|__MINUTES__|$MINUTES|g" \
    -e "s|__AUTORUN__|$AUTORUN|g" -e "s|__MODE__|$MODE|g" -e "s|__BUCKET__|$BUCKET|g" -e "s|__DEPLOY_KEY_B64__|$KEY_B64|g" \
    -e "s|__GIT_NAME__|$(git config user.name)|g" -e "s|__GIT_EMAIL__|$(git config user.email)|g" \
    "$ROOT/infra/user-data.tpl.sh" > "$STATE/user-data.sh"

if date -v+1M >/dev/null 2>&1; then EXPIRES=$(date -u -v+"${MINUTES}"M +%FT%TZ)
else EXPIRES=$(date -u -d "+$MINUTES min" +%FT%TZ); fi

say "Launching $ITYPE, MODE=$MODE (trying default subnets in $AZS until one has capacity)"
IID=""
for SUBNET in $SUBNETS; do
  if IID=$($AWS ec2 run-instances \
      --image-id "$AMI" --instance-type "$ITYPE" --key-name "$NAME" \
      --subnet-id "$SUBNET" --security-group-ids "$SG" \
      --instance-initiated-shutdown-behavior terminate --iam-instance-profile Name="$PROFILE" \
      --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"VolumeSize\":$ROOT_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME-$MODE},{Key=expires-at,Value=$EXPIRES},{Key=repo,Value=$REPO}]" \
      --user-data "file://$STATE/user-data.sh" \
      --query 'Instances[0].InstanceId' --output text 2>"$STATE/run-err.txt"); then
    break
  fi
  echo "  $SUBNET: $(tail -1 "$STATE/run-err.txt")"; IID=""
done
if [ -z "$IID" ]; then
  if grep -q VcpuLimitExceeded "$STATE/run-err.txt"; then
    echo "vCPU quota exhausted (not an AWS capacity problem). Check running Trn instances or request more quota."
  else
    echo "No capacity for $ITYPE in $REGION right now. Retry later or pick another region from ./infra/regions.sh"
  fi
  exit 1
fi

printf 'IID=%s\nREGION=%s\nREPO=%s\nKEY_TITLE=%s\n' "$IID" "$REGION" "$REPO" "$KEY_TITLE" > "$STATE/instance-$MODE.env"
$AWS ec2 wait instance-running --instance-ids "$IID"
DNS=$($AWS ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

say "Up: $IID (auto-terminates ~$EXPIRES)"
cat <<EOF
  ssh -i $STATE/login_key ubuntu@$DNS
  ssh -i $STATE/login_key ubuntu@$DNS tail -f /var/log/cloud-init-output.log   # boot + setup log
  ssh -i $STATE/login_key ubuntu@$DNS tail -f work/results/latest/console.log  # benchmark log
  git pull                                                                      # results land here every 10 min
  ./infra/teardown.sh                                                           # when done (or after auto-termination)
EOF
