#!/usr/bin/env bash
# Idempotently create (or reuse) an isolated lab network, so the lab never depends on how the
# account's default VPC has been configured:
#   VPC 10.42.0.0/16 -> internet gateway -> public route table
#   one public /24 subnet per AZ that offers $ITYPE (CIDR 10.42.<AZ letter index>.0/24)
#   security group: SSH from every public IP this machine has been seen with (stored in .state/ssh_cidrs,
#   since egress IPs often rotate) plus any extra SSH_CIDRS="1.2.3.4/32 ..."; anything else is revoked.
# A new VPC's default network ACL allows all traffic; the security group does the filtering.
# Writes infra/.state/network.env with VPC, SG, SUBNETS.
set -euo pipefail

REGION=${REGION:-us-east-1}
ITYPE=${ITYPE:-trn1.2xlarge}
NAME=${NAME:-neuron-lab}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STATE=$ROOT/infra/.state
mkdir -p "$STATE"
AWS="aws --region $REGION"
tags() { echo "ResourceType=$1,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=$NAME}]"; }
log() { echo "  $*" >&2; }

VPC=$($AWS ec2 describe-vpcs --filters Name=tag:Name,Values="$NAME" --query 'Vpcs[0].VpcId' --output text)
if [ "$VPC" = "None" ]; then
  VPC=$($AWS ec2 create-vpc --cidr-block 10.42.0.0/16 --tag-specifications "$(tags vpc)" --query Vpc.VpcId --output text)
  $AWS ec2 wait vpc-available --vpc-ids "$VPC"
  $AWS ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-support '{"Value":true}'
  $AWS ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-hostnames '{"Value":true}'
  log "created VPC $VPC"
fi

IGW=$($AWS ec2 describe-internet-gateways --filters Name=attachment.vpc-id,Values="$VPC" \
  --query 'InternetGateways[0].InternetGatewayId' --output text)
if [ "$IGW" = "None" ]; then
  IGW=$($AWS ec2 create-internet-gateway --tag-specifications "$(tags internet-gateway)" \
    --query InternetGateway.InternetGatewayId --output text)
  $AWS ec2 attach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC"
  log "created internet gateway $IGW"
fi

RT=$($AWS ec2 describe-route-tables --filters Name=vpc-id,Values="$VPC" Name=tag:Name,Values="$NAME" \
  --query 'RouteTables[0].RouteTableId' --output text)
if [ "$RT" = "None" ]; then
  RT=$($AWS ec2 create-route-table --vpc-id "$VPC" --tag-specifications "$(tags route-table)" \
    --query RouteTable.RouteTableId --output text)
  $AWS ec2 create-route --route-table-id "$RT" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW" >/dev/null
  log "created route table $RT"
fi

SUBNETS=""
for AZ in $($AWS ec2 describe-instance-type-offerings --location-type availability-zone \
              --filters Name=instance-type,Values="$ITYPE" --query 'InstanceTypeOfferings[].Location' --output text); do
  SN=$($AWS ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" Name=availability-zone,Values="$AZ" \
    --query 'Subnets[0].SubnetId' --output text)
  if [ "$SN" = "None" ]; then
    IDX=$(( $(printf '%d' "'${AZ: -1}") - 96 ))   # us-east-1c -> 3 -> 10.42.3.0/24
    SN=$($AWS ec2 create-subnet --vpc-id "$VPC" --availability-zone "$AZ" --cidr-block "10.42.$IDX.0/24" \
      --tag-specifications "$(tags subnet)" --query Subnet.SubnetId --output text)
    $AWS ec2 modify-subnet-attribute --subnet-id "$SN" --map-public-ip-on-launch
    $AWS ec2 associate-route-table --route-table-id "$RT" --subnet-id "$SN" >/dev/null
    log "created subnet $SN in $AZ"
  fi
  SUBNETS="$SUBNETS $SN"
done
[ -n "$SUBNETS" ] || { echo "$ITYPE is not offered in $REGION. Run ./infra/regions.sh" >&2; exit 1; }

SG=$($AWS ec2 describe-security-groups --filters Name=group-name,Values="$NAME-ssh" Name=vpc-id,Values="$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$($AWS ec2 create-security-group --group-name "$NAME-ssh" --description "SSH for $NAME" --vpc-id "$VPC" \
    --tag-specifications "$(tags security-group)" --query GroupId --output text)
  log "created security group $SG"
fi
# Remember every IP we've launched from (egress IPs often rotate); RESET_SSH=1 starts over.
CIDR_FILE=$STATE/ssh_cidrs
[ "${RESET_SSH:-0}" = 1 ] && rm -f "$CIDR_FILE"
SSH_CIDRS=$( { cat "$CIDR_FILE" 2>/dev/null; echo "${SSH_CIDRS:-}" | tr ' ' '\n';
               for _ in 1 2 3; do echo "$(curl -fsS https://checkip.amazonaws.com)/32"; done; } | grep -E '^[0-9./]+$' | sort -u)
echo "$SSH_CIDRS" > "$CIDR_FILE"
SSH_CIDRS=$(echo $SSH_CIDRS)
for CIDR in $($AWS ec2 describe-security-groups --group-ids "$SG" \
                --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text); do
  case " $SSH_CIDRS " in *" $CIDR "*) ;; *)
    $AWS ec2 revoke-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$CIDR" >/dev/null
    log "revoked SSH from $CIDR" ;;
  esac
done
for CIDR in $SSH_CIDRS; do
  $AWS ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$CIDR" >/dev/null 2>&1 &&
    log "allowed SSH from $CIDR" || true
done

printf 'VPC=%s\nSG=%s\nSUBNETS="%s"\n' "$VPC" "$SG" "${SUBNETS# }" | tee "$STATE/network.env" >&2
