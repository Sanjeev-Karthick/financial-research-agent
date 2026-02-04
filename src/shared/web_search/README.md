# Web Search Lambda Tool

This Lambda function provides web search capabilities using the [Tavily API](https://tavily.com/) for retrieving real-time financial news and market information.

## Prerequisites

1. **Tavily API Key**: Sign up at [Tavily](https://tavily.com/) and get your API key
2. **AWS SAM CLI**: Install the [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)

## Deployment

1. Set your Tavily API key as an environment variable or update the template:

```bash
export TAVILY_API_KEY="your-api-key-here"
```

2. Build and deploy using SAM:

```bash
cd src/shared/web_search
sam build
sam deploy --guided
```

3. Note the Lambda function ARN from the output - you'll need this for the agent configuration.

## Usage

The Lambda function accepts the following input parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `search_query` | string | Yes | The search query to execute |
| `target_website` | string | No | Specific website to search (e.g., "reuters.com") |
| `topic` | string | No | Topic category (e.g., "news", "finance") |
| `days` | string | No | Number of days of history to search |

## Example Input

```json
{
  "search_query": "AAPL earnings report Q4 2024",
  "topic": "news",
  "days": "7"
}
```

## Example Output

```json
{
  "results": [
    {
      "title": "Apple Reports Q4 2024 Earnings...",
      "url": "https://...",
      "content": "Apple Inc. reported...",
      "score": 0.95
    }
  ]
}
```

## IAM Permissions

The Lambda function needs permission to:
- Access Secrets Manager (if storing API key there)
- CloudWatch Logs for logging

## Cost Considerations

- Tavily API has usage-based pricing
- Lambda invocations incur standard AWS Lambda costs
- Consider implementing caching for frequently requested queries
