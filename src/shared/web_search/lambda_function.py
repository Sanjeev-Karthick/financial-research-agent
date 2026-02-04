"""
Web Search Lambda Function

This Lambda function provides web search capabilities for Amazon Bedrock Agents
using the Tavily API. It's designed to retrieve real-time financial news,
market updates, and company information.

The function is invoked by Bedrock agents through action groups and returns
search results in a format that agents can understand and process.

Environment Variables:
    TAVILY_API_KEY: Your Tavily API key (required)
    
Author: Financial Research Agent Team
License: Apache-2.0
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode

# Set up logging - Lambda uses CloudWatch Logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Tavily API configuration
TAVILY_API_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_SEARCH_DEPTH = "basic"  # "basic" or "advanced"
MAX_RESULTS = 5


def get_api_key() -> str:
    """
    Retrieve the Tavily API key from environment variables.
    
    In production, you might want to use AWS Secrets Manager instead
    for better security practices.
    
    Returns:
        The Tavily API key
        
    Raises:
        ValueError: If the API key is not configured
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    
    if not api_key:
        logger.error("TAVILY_API_KEY environment variable is not set")
        raise ValueError("Tavily API key is not configured")
    
    return api_key


def search_web(
    query: str,
    target_website: Optional[str] = None,
    topic: Optional[str] = None,
    days: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute a web search using the Tavily API.
    
    This function queries Tavily's search API and returns relevant
    results including titles, URLs, and content snippets.
    
    Args:
        query: The search query string
        target_website: Optional specific domain to search
        topic: Optional topic category (e.g., "news", "finance")
        days: Optional number of days of history to search
        
    Returns:
        Dictionary containing search results and metadata
    """
    api_key = get_api_key()
    
    # Build the request payload
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": DEFAULT_SEARCH_DEPTH,
        "max_results": MAX_RESULTS,
        "include_answer": True,
        "include_raw_content": False,
    }
    
    # Add optional parameters if provided
    if target_website:
        payload["include_domains"] = [target_website]
        
    if topic:
        payload["topic"] = topic
        
    if days:
        # Tavily uses 'd' suffix for days, e.g., "7d"
        payload["days"] = str(days)
    
    logger.info(f"Executing web search for query: {query}")
    
    try:
        # Make the API request
        request = Request(
            TAVILY_API_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            method="POST"
        )
        
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            
        logger.info(f"Search returned {len(result.get('results', []))} results")
        
        return {
            "success": True,
            "query": query,
            "answer": result.get("answer", ""),
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "published_date": r.get("published_date", "")
                }
                for r in result.get("results", [])
            ]
        }
        
    except HTTPError as e:
        logger.error(f"HTTP error during search: {e.code} - {e.reason}")
        return {
            "success": False,
            "error": f"Search API error: {e.reason}",
            "results": []
        }
        
    except URLError as e:
        logger.error(f"URL error during search: {e.reason}")
        return {
            "success": False,
            "error": f"Network error: {e.reason}",
            "results": []
        }
        
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


def parse_agent_input(event: Dict) -> Dict[str, Any]:
    """
    Parse the input from a Bedrock Agent action group invocation.
    
    Bedrock Agents send inputs in a specific format that needs to be
    extracted and converted to usable parameters.
    
    Args:
        event: The Lambda event from Bedrock Agent
        
    Returns:
        Dictionary of parsed parameters
    """
    parameters = {}
    
    # Bedrock Agent format - parameters come in requestBody or parameters field
    if "requestBody" in event:
        content = event["requestBody"].get("content", {})
        if "application/json" in content:
            body = content["application/json"]
            if "properties" in body:
                for prop in body["properties"]:
                    parameters[prop["name"]] = prop["value"]
    
    # Alternative format - direct parameters list
    if "parameters" in event:
        for param in event["parameters"]:
            parameters[param["name"]] = param["value"]
    
    # Also check for direct key access (for testing)
    for key in ["search_query", "target_website", "topic", "days"]:
        if key in event and key not in parameters:
            parameters[key] = event[key]
    
    return parameters


def format_agent_response(
    action_group: str,
    function: str,
    result: Dict
) -> Dict:
    """
    Format the response for Bedrock Agent consumption.
    
    Bedrock Agents expect responses in a specific format with
    action group and function information.
    
    Args:
        action_group: Name of the action group
        function: Name of the function called
        result: The actual result to return
        
    Returns:
        Properly formatted response dictionary
    """
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "function": function,
            "functionResponse": {
                "responseBody": {
                    "TEXT": {
                        "body": json.dumps(result)
                    }
                }
            }
        }
    }


def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    Main Lambda handler for web search requests.
    
    This function is invoked by Amazon Bedrock Agents when the
    web_search action group is called. It parses the input,
    executes the search, and returns formatted results.
    
    Args:
        event: Lambda event containing search parameters
        context: Lambda context object
        
    Returns:
        Formatted response for Bedrock Agent
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Extract action group and function info
    action_group = event.get("actionGroup", "web_search")
    function = event.get("function", "web_search")
    
    try:
        # Parse the input parameters
        params = parse_agent_input(event)
        
        # Validate required parameters
        search_query = params.get("search_query")
        if not search_query:
            raise ValueError("search_query parameter is required")
        
        # Extract optional parameters
        target_website = params.get("target_website")
        topic = params.get("topic")
        days = params.get("days")
        
        # Convert days to integer if provided
        if days:
            try:
                days = int(days)
            except ValueError:
                days = None
        
        # Execute the search
        result = search_web(
            query=search_query,
            target_website=target_website,
            topic=topic,
            days=days
        )
        
        # Format and return the response
        return format_agent_response(action_group, function, result)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        
        error_result = {
            "success": False,
            "error": str(e),
            "results": []
        }
        
        return format_agent_response(action_group, function, error_result)


# For local testing
if __name__ == "__main__":
    # Test event simulating Bedrock Agent input
    test_event = {
        "actionGroup": "web_search",
        "function": "web_search",
        "parameters": [
            {"name": "search_query", "value": "AAPL stock news today"},
            {"name": "topic", "value": "news"},
            {"name": "days", "value": "7"}
        ]
    }
    
    # Make sure to set TAVILY_API_KEY environment variable before testing
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
