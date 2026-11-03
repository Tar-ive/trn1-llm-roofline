#!/usr/bin/env bash
# Delete the lab network created by infra/network.sh. Fails safely if instances are still in it.
set -euo pipefail
REGION=${REGION:-us-east-1}
NAME=${NAME:-neuron-lab}
AWS="aws --region $REGION"

VPC=$($AWS ec2 describe-vpcs --filters Name=tag:Name,Values="$NAME" --query 'Vpcs[0].VpcId' --output text)
[ "$VPC" = "None" ] && { echo "No $NAME VPC in $REGION"; exit 0; }
LIVE=$($AWS ec2 describe-instances --filters Name=vpc-id,Values="$VPC" \
  Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down --query 'Reservations[].Instances[].InstanceId' --output text)
[ -z "$LIVE" ] || { echo "Instances still in $VPC: $LIVE"; exit 1; }

for SN in $($AWS ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" --query 'Subnets[].SubnetId' --output text); do
  $AWS ec2 delete-subnet --subnet-id "$SN"
done
for RT in $($AWS ec2 describe-route-tables --filters Name=vpc-id,Values="$VPC" Name=tag:Name,Values="$NAME" --query 'RouteTables[].RouteTableId' --output text); do
  $AWS ec2 delete-route-table --route-table-id "$RT"
done
for IGW in $($AWS ec2 describe-internet-gateways --filters Name=attachment.vpc-id,Values="$VPC" --query 'InternetGateways[].InternetGatewayId' --output text); do
  $AWS ec2 detach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC"
  $AWS ec2 delete-internet-gateway --internet-gateway-id "$IGW"
done
for SG in $($AWS ec2 describe-security-groups --filters Name=vpc-id,Values="$VPC" Name=group-name,Values="$NAME-ssh" --query 'SecurityGroups[].GroupId' --output text); do
  $AWS ec2 delete-security-group --group-id "$SG"
done
$AWS ec2 delete-vpc --vpc-id "$VPC"
echo "Deleted $VPC and its subnets, route table, internet gateway and security group"
