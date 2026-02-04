# Stock Data Lambda Tool

This Lambda function provides stock data lookup and portfolio optimization capabilities for the Financial Research Agent.

## Features

- **Stock Price Lookup**: Get historical stock prices for any ticker symbol
- **Portfolio Optimization**: Mean-variance optimization for multi-asset portfolios

## Prerequisites

1. **AWS SAM CLI**: Install the [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
2. **Python Dependencies**: The function uses `yfinance` for stock data and `scipy` for optimization

## Deployment

Build and deploy using SAM:

```bash
cd src/shared/stock_data
sam build
sam deploy --guided
```

> **Note**: This stack may take ~5 minutes to deploy due to the Lambda layer compilation.

## Functions

### stock_data_lookup

Retrieves 1-month historical stock prices for a given ticker.

**Input Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ticker` | string | Yes | Stock ticker symbol (e.g., "AAPL", "MSFT") |

**Example Input:**

```json
{
  "ticker": "AAPL"
}
```

**Example Output:**

```json
{
  "ticker": "AAPL",
  "prices": {
    "2024-01-15": 185.92,
    "2024-01-16": 183.63,
    "2024-01-17": 182.68
  },
  "period": "1 month"
}
```

### portfolio_optimization

Optimizes portfolio allocation using mean-variance optimization (Modern Portfolio Theory).

**Input Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tickers` | string | Yes | Comma-separated list of stock tickers (minimum 3) |
| `prices` | string | Yes | JSON object with historical prices |

**Example Input:**

```json
{
  "tickers": "AAPL,MSFT,GOOGL",
  "prices": "{\"2024-01-15\": {\"AAPL\": 185.92, \"MSFT\": 390.53, \"GOOGL\": 140.46}}"
}
```

**Example Output:**

```json
{
  "allocations": {
    "AAPL": 0.35,
    "MSFT": 0.45,
    "GOOGL": 0.20
  },
  "expected_return": 0.12,
  "volatility": 0.18,
  "sharpe_ratio": 0.67
}
```

## Architecture

```
┌──────────────────────────────────────┐
│          Bedrock Agent               │
│   (Quantitative Analysis Agent)      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│    Stock Data Lambda Function        │
├──────────────────────────────────────┤
│  • stock_data_lookup()               │
│  • portfolio_optimization()          │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         Yahoo Finance API            │
│   (via yfinance Python library)      │
└──────────────────────────────────────┘
```

## IAM Permissions

The Lambda function needs:
- Basic Lambda execution permissions
- CloudWatch Logs access

## Cost Considerations

- Lambda invocation costs
- No external API costs (yfinance uses free Yahoo Finance data)

## Limitations

- Yahoo Finance data may have slight delays
- Portfolio optimization requires at least 3 tickers
- Historical data availability depends on ticker age
