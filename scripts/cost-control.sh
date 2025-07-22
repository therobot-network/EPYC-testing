#!/bin/bash
# Manual Cost Control Script for EPYC-testing
# Provides immediate cost control capabilities

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
INSTANCE_ID="i-00268dae9fd36421f"
REGION="us-west-1"

echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              EPYC-testing Cost Control                      ║"
echo "║            Manual Instance Management                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check AWS profile
if [ -z "$AWS_PROFILE" ]; then
    echo -e "${YELLOW}⚠️  Setting AWS profile to epyc-testing${NC}"
    export AWS_PROFILE=epyc-testing
fi

# Function to get instance status
get_instance_status() {
    aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region $REGION \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text
}

# Function to get instance cost estimate
get_cost_estimate() {
    local hours_running=$1
    local hourly_rate=4.608  # c6a.24xlarge on-demand rate (approximate)
    
    if command -v bc >/dev/null 2>&1; then
        local daily_cost=$(echo "scale=2; $hourly_rate * 24" | bc)
        local current_cost=$(echo "scale=2; $hourly_rate * $hours_running" | bc)
    else
        # Fallback using shell arithmetic (less precise)
        local daily_cost=$(( hourly_rate * 24 ))  # Will truncate decimals
        local current_cost=$(( hourly_rate * hours_running ))
    fi
    
    echo -e "${BLUE}💰 Cost Estimates:${NC}"
    echo -e "   Hourly rate: \$${hourly_rate}"
    echo -e "   Daily cost: \$${daily_cost}"
    echo -e "   Current session: \$${current_cost}"
}

# Function to stop instance
stop_instance() {
    echo -e "${YELLOW}🛑 Stopping instance $INSTANCE_ID...${NC}"
    aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION
    echo -e "${GREEN}✅ Stop command sent. Instance will shut down shortly.${NC}"
}

# Function to start instance
start_instance() {
    echo -e "${YELLOW}🚀 Starting instance $INSTANCE_ID...${NC}"
    aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION
    echo -e "${GREEN}✅ Start command sent. Instance will boot up shortly.${NC}"
}

# Function to get instance metrics
get_metrics() {
    local end_time=$(date -u +"%Y-%m-%dT%H:%M:%S")
    local start_time=$(date -u -v-1H +"%Y-%m-%dT%H:%M:%S")
    
    echo -e "${BLUE}📊 Instance Metrics (last hour):${NC}"
    
    # Get CPU utilization
    local cpu_avg=$(aws cloudwatch get-metric-statistics \
        --namespace AWS/EC2 \
        --metric-name CPUUtilization \
        --dimensions Name=InstanceId,Value=$INSTANCE_ID \
        --start-time $start_time \
        --end-time $end_time \
        --period 3600 \
        --statistics Average \
        --region $REGION \
        --query 'Datapoints[0].Average' \
        --output text 2>/dev/null || echo "N/A")
    
    if [ "$cpu_avg" != "None" ] && [ "$cpu_avg" != "N/A" ]; then
        echo -e "   CPU Average: ${cpu_avg}%"
    else
        echo -e "   CPU Average: No data (instance may be stopped or recently started)"
    fi
}

# Main menu
show_menu() {
    echo -e "${YELLOW}📋 Available Actions:${NC}"
    echo "1. Check instance status"
    echo "2. View cost estimates"
    echo "3. View instance metrics"
    echo "4. Stop instance (save costs)"
    echo "5. Start instance"
    echo "6. Emergency stop (immediate)"
    echo "7. Exit"
    echo ""
}

# Main loop
while true; do
    current_status=$(get_instance_status)
    echo -e "${BLUE}🔍 Current Status: ${BOLD}$current_status${NC}"
    echo ""
    
    show_menu
    read -p "Choose an action (1-7): " choice
    echo ""
    
    case $choice in
        1)
            echo -e "${GREEN}✅ Instance Status: $current_status${NC}"
            if [ "$current_status" = "running" ]; then
                # Get launch time for cost calculation
                launch_time=$(aws ec2 describe-instances \
                    --instance-ids $INSTANCE_ID \
                    --region $REGION \
                    --query 'Reservations[0].Instances[0].LaunchTime' \
                    --output text)
                echo -e "   Launch time: $launch_time"
            fi
            ;;
        2)
            if [ "$current_status" = "running" ]; then
                # Calculate hours running  
                launch_time=$(aws ec2 describe-instances \
                    --instance-ids $INSTANCE_ID \
                    --region $REGION \
                    --query 'Reservations[0].Instances[0].LaunchTime' \
                    --output text)
                current_time=$(date -u +%s)
                # Parse AWS timestamp format (works with both .000Z and Z endings)
                clean_time=$(echo "$launch_time" | sed 's/\.[0-9]*Z$/Z/')
                launch_timestamp=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$clean_time" +%s 2>/dev/null || echo "0")
                if command -v bc >/dev/null 2>&1; then
                    hours_running=$(echo "scale=2; ($current_time - $launch_timestamp) / 3600" | bc)
                else
                    hours_running=$(( (current_time - launch_timestamp) / 3600 ))
                fi
                get_cost_estimate $hours_running
            else
                echo -e "${YELLOW}💰 Instance is not running - no current costs${NC}"
                get_cost_estimate 0
            fi
            ;;
        3)
            get_metrics
            ;;
        4)
            if [ "$current_status" = "running" ]; then
                echo -e "${YELLOW}⚠️  Are you sure you want to stop the instance? (y/N)${NC}"
                read -p "Confirm: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    stop_instance
                else
                    echo -e "${BLUE}ℹ️  Operation cancelled${NC}"
                fi
            else
                echo -e "${YELLOW}ℹ️  Instance is already stopped${NC}"
            fi
            ;;
        5)
            if [ "$current_status" = "stopped" ]; then
                echo -e "${YELLOW}⚠️  Starting instance will incur costs. Continue? (y/N)${NC}"
                read -p "Confirm: " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    start_instance
                else
                    echo -e "${BLUE}ℹ️  Operation cancelled${NC}"
                fi
            else
                echo -e "${YELLOW}ℹ️  Instance is not stopped (current: $current_status)${NC}"
            fi
            ;;
        6)
            echo -e "${RED}🚨 EMERGENCY STOP - This will immediately stop the instance${NC}"
            echo -e "${YELLOW}⚠️  Are you absolutely sure? (type 'STOP' to confirm)${NC}"
            read -p "Confirm: " confirm
            if [ "$confirm" = "STOP" ]; then
                stop_instance
                echo -e "${RED}🚨 Emergency stop initiated${NC}"
            else
                echo -e "${BLUE}ℹ️  Emergency stop cancelled${NC}"
            fi
            ;;
        7)
            echo -e "${GREEN}👋 Goodbye! Remember to stop your instance when not in use.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Invalid option. Please choose 1-7.${NC}"
            ;;
    esac
    
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
    clear
done 