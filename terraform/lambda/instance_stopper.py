import boto3
import json
import os
from datetime import datetime, timedelta

def handler(event, context):
    """
    Lambda function to stop EC2 instance based on cost thresholds
    """
    
    # Initialize AWS clients
    ec2 = boto3.client('ec2')
    sns = boto3.client('sns')
    cloudwatch = boto3.client('cloudwatch')
    
    instance_id = os.environ['INSTANCE_ID']
    
    try:
        # Check if instance is running
        response = ec2.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        current_state = instance['State']['Name']
        
        if current_state != 'running':
            print(f"Instance {instance_id} is not running (state: {current_state})")
            return {
                'statusCode': 200,
                'body': json.dumps(f'Instance is not running: {current_state}')
            }
        
        # Check CPU utilization over the last hour
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        cpu_response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': instance_id
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,  # 5 minutes
            Statistics=['Average']
        )
        
        # Calculate average CPU over the last hour
        if cpu_response['Datapoints']:
            avg_cpu = sum(point['Average'] for point in cpu_response['Datapoints']) / len(cpu_response['Datapoints'])
        else:
            avg_cpu = 0
        
        # Check network out over the last hour
        network_response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='NetworkOut',
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': instance_id
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Sum']
        )
        
        # Calculate total network out
        total_network_out = 0
        if network_response['Datapoints']:
            total_network_out = sum(point['Sum'] for point in network_response['Datapoints'])
        
        # Decision logic
        should_stop = False
        reason = ""
        
        # Stop if CPU is very low for extended period (likely idle)
        if avg_cpu < 5:
            should_stop = True
            reason = f"Low CPU utilization ({avg_cpu:.2f}%) - instance appears idle"
        
        # Stop if very high network usage (potential runaway process)
        elif total_network_out > 50 * 1024 * 1024 * 1024:  # 50 GB
            should_stop = True
            reason = f"High network usage ({total_network_out / (1024**3):.2f} GB) - potential cost issue"
        
        # Create message
        message = f"""
EPYC-testing Instance Status Report
Instance ID: {instance_id}
Current State: {current_state}
Average CPU (1h): {avg_cpu:.2f}%
Network Out (1h): {total_network_out / (1024**2):.2f} MB

Action: {'STOPPING INSTANCE' if should_stop else 'No action needed'}
Reason: {reason if should_stop else 'Instance metrics within normal ranges'}

Time: {datetime.utcnow().isoformat()} UTC
        """
        
        # Send notification
        sns_topic_arn = f"arn:aws:sns:{boto3.Session().region_name}:{context.invoked_function_arn.split(':')[4]}:epyc-testing-cost-alerts"
        
        try:
            sns.publish(
                TopicArn=sns_topic_arn,
                Subject=f"EPYC-testing Instance {'STOPPED' if should_stop else 'Status'} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                Message=message
            )
        except Exception as e:
            print(f"Failed to send SNS notification: {e}")
        
        # Stop instance if needed
        if should_stop:
            print(f"Stopping instance {instance_id}: {reason}")
            ec2.stop_instances(InstanceIds=[instance_id])
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'action': 'stopped',
                    'reason': reason,
                    'instance_id': instance_id
                })
            }
        else:
            print(f"Instance {instance_id} metrics are normal, no action needed")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'action': 'none',
                    'reason': 'Metrics within normal ranges',
                    'instance_id': instance_id,
                    'cpu_avg': avg_cpu,
                    'network_out_mb': total_network_out / (1024**2)
                })
            }
            
    except Exception as e:
        error_msg = f"Error processing instance {instance_id}: {str(e)}"
        print(error_msg)
        
        # Send error notification
        try:
            sns.publish(
                TopicArn=sns_topic_arn,
                Subject="EPYC-testing Cost Control Error",
                Message=f"Error in cost control Lambda function:\n\n{error_msg}\n\nTime: {datetime.utcnow().isoformat()} UTC"
            )
        except:
            pass
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        } 