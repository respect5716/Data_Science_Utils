
def send_by_slack(token: str, channel: str, message: str):
    import slack
    client = slack.WebClient(token = token)
    response = client.chat_postMessage(channel = channel, text = message)
    return response


def send_by_aws_sns(access_key_id: str, secrete_access_key: str, region_name: str, topic_arn: str, message: str):
    import boto3
    client = boto3.client('sns',
        aws_access_key_id = access_key_id,
        aws_secret_access_key = secrete_access_key,
        region_name = region_name,
    )
    
    response = client.publish(
        TopicArn = topic_arn,
        Message = message
    )
    
    return response
