# Financial Research Agent Example

This example demonstrates how to create a multi-agent collaboration system for financial research using Amazon Bedrock.

## Overview

The Financial Research Assistant uses a supervisor agent pattern to orchestrate three specialized sub-agents:

1. **News Agent**: Retrieves and analyzes financial news and documents
2. **Quantitative Analysis Agent**: Fetches stock data and performs portfolio optimization
3. **Smart Summarizer Agent**: Synthesizes insights into structured investment reports

## Prerequisites

Before running this notebook, ensure you have:

1. Completed the setup from the main [README](../../../README.md)
2. Deployed the Web Search Lambda tool
3. Deployed the Stock Data Lambda tool
4. Enabled required foundation models in Amazon Bedrock

## Running the Notebook

```bash
jupyter notebook main.ipynb
```

## Example Queries

- "What's AAPL stock price doing over the last week and relate that to recent news"
- "Optimize my portfolio with AAPL, MSFT, and GOOGL"
- "Analyze Amazon's financial health based on recent earnings reports"

## Architecture

```
┌────────────────────────────────────────┐
│    Financial Research Supervisor       │
└──────────────────┬─────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐  ┌───────────┐  ┌───────────┐
│  News  │  │ Quant     │  │  Smart    │
│ Agent  │  │ Analysis  │  │Summarizer │
└────────┘  └───────────┘  └───────────┘
```

> [!NOTE]
> Results from these agents should not be taken as financial advice.
