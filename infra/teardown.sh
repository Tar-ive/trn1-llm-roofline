#!/usr/bin/env bash
# Remove the ephemeral deploy key, make sure the instance is gone, pull results.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${MODE:-run}   # MODE=compile ./infra/teardown.sh for the compile box
source "$ROOT/infra/.state/instance-$MODE.env"

STATE=$(aws --region "$REGION" ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo gone)
echo "Instance $IID: $STATE"
if [ "$STATE" != "terminated" ] && [ "$STATE" != "gone" ]; then
  read -rp "Terminate now? Unpushed work on the box is lost unless you ssh in and run 'autosave'. [y/N] " a
  [ "$a" = y ] && aws --region "$REGION" ec2 terminate-instances --instance-ids "$IID" >/dev/null
fi

ID=$(gh repo deploy-key list --repo "$REPO" --json id,title -q ".[] | select(.title==\"$KEY_TITLE\") | .id")
[ -n "$ID" ] && gh repo deploy-key delete "$ID" --repo "$REPO" && echo "Deleted deploy key $KEY_TITLE"
rm -f "$ROOT/infra/.state/deploy_key" "$ROOT/infra/.state/deploy_key.pub"

git -C "$ROOT" pull --rebase
