#!/usr/bin/env bash
# Where can I launch $ITYPE right now? For each region: AZs that offer the type, and the
# on-demand + spot vCPU quota for its family. Quotas are per region and start at 0 on new accounts.
#
#   ./infra/regions.sh                         # all enabled regions
#   ITYPE=trn2.48xlarge ./infra/regions.sh us-east-1 us-east-2
set -uo pipefail
ITYPE=${ITYPE:-trn1.2xlarge}
OD_QUOTA=${OD_QUOTA:-L-2C3B7624}     # Running On-Demand Trn instances (vCPUs)
SPOT_QUOTA=${SPOT_QUOTA:-L-6B0D517C} # All Trn Spot Instance Requests (vCPUs)

REGIONS=${*:-$(aws ec2 describe-regions --region us-east-1 --query 'Regions[].RegionName' --output text)}
VCPUS=$(aws ec2 describe-instance-types --region us-east-1 --instance-types "$ITYPE" \
  --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text 2>/dev/null || echo "?")

printf '%-16s %-34s %8s %8s  %s\n' REGION "AZS OFFERING $ITYPE" ON-DEM SPOT VERDICT
for r in $REGIONS; do
  azs=$(aws ec2 describe-instance-type-offerings --region "$r" --location-type availability-zone \
    --filters Name=instance-type,Values="$ITYPE" --query 'InstanceTypeOfferings[].Location' --output text 2>/dev/null)
  [ -z "$azs" ] && continue
  q() { aws service-quotas get-service-quota --region "$r" --service-code ec2 --quota-code "$1" \
          --query 'Quota.Value' --output text 2>/dev/null | cut -d. -f1; }
  od=$(q "$OD_QUOTA"); spot=$(q "$SPOT_QUOTA")
  verdict="needs quota increase"
  [ "${od:-0}" -ge "$VCPUS" ] 2>/dev/null && verdict="OK on-demand"
  [ "$verdict" != "OK on-demand" ] && [ "${spot:-0}" -ge "$VCPUS" ] 2>/dev/null && verdict="OK spot only"
  printf '%-16s %-34s %8s %8s  %s\n' "$r" "$(echo $azs | tr ' ' ',')" "${od:-?}" "${spot:-?}" "$verdict"
done
echo
echo "$ITYPE needs $VCPUS vCPUs. To request on-demand quota in a region:"
echo "  aws service-quotas request-service-quota-increase --region <region> --service-code ec2 --quota-code $OD_QUOTA --desired-value $VCPUS"
