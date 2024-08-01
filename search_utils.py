# search_utils.py
import os
import json
import logging
import asyncio
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Dict, Any


load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "SEARXNG")
SEARXNG_URL = os.getenv("SEARXNG_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEARCH_RESULTS_LIMIT = int(os.getenv("SEARCH_RESULTS_LIMIT", 5))

# Initialize Tavily client if API key is provided
if TAVILY_API_KEY:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

async def perform_search(query: str) -> Dict[str, Any]:
    """
    Perform a search using the configured search provider.
    """
    if SEARCH_PROVIDER == "SEARXNG":
        return await searxng_search(query)
    elif SEARCH_PROVIDER == "TAVILY":
        return await tavily_search(query)
    else:
        return {"success": False, "error": f"Unknown search provider '{SEARCH_PROVIDER}'"}

async def searxng_search(query: str) -> Dict[str, Any]:
    """
    Perform a search using the local SearXNG instance.
    """
    params = {
        "q": query,
        "format": "json"
    }
    headers = {
        "User-Agent": "OllamaAssistant/1.0"
    }
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(SEARXNG_URL, params=params, headers=headers, timeout=30)
        )
        response.raise_for_status()
        results = response.json()
        
        formatted_results = []
        for result in results.get('results', [])[:SEARCH_RESULTS_LIMIT]:
            formatted_results.append({
                "title": result['title'],
                "url": result['url'],
                "snippet": result.get('content', 'No snippet available')
            })
        
        return {"success": True, "results": formatted_results}
    except requests.RequestException as e:
        return {"success": False, "error": f"Error performing SearXNG search: {str(e)}"}



async def tavily_search(query: str) -> Dict[str, Any]:
    """
    Perform a search using Tavily.
    """
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: tavily_client.get_search_context(query, search_depth="advanced", max_results=SEARCH_RESULTS_LIMIT)
        )
        
        logger.debug(f"Tavily raw response: {response}")
        
        if isinstance(response, str):
            try:
                results = json.loads(response)
            except json.JSONDecodeError:
                results = [response]
        elif isinstance(response, (list, dict)):
            results = response if isinstance(response, list) else [response]
        else:
            results = [response]
        
        # If results are individual characters, join them
        if all(isinstance(r, str) and len(r) == 1 for r in results):
            joined_text = ''.join(results)
            try:
                parsed_json = json.loads(joined_text)
                if isinstance(parsed_json, list):
                    results = parsed_json
                else:
                    results = [parsed_json]
            except json.JSONDecodeError:
                results = [joined_text]
        
        formatted_results = []
        for result in results:
            if isinstance(result, (int, float)):
                formatted_results.append({
                    "title": "Numeric result",
                    "url": "",
                    "snippet": str(result)
                })
            elif isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    url = result_dict.get('url', 'No URL')
                    content = result_dict.get('content', 'No content')
                    title = result_dict.get('title', urlparse(url).netloc or "No title")
                    formatted_results.append({
                        "title": title,
                        "url": url,
                        "snippet": content
                    })
                except json.JSONDecodeError:
                    formatted_results.append({
                        "title": "Text result",
                        "url": "",
                        "snippet": result
                    })
            elif isinstance(result, dict):
                url = result.get('url', 'No URL')
                content = result.get('content', 'No content')
                title = result.get('title', urlparse(url).netloc or "No title")
                formatted_results.append({
                    "title": title,
                    "url": url,
                    "snippet": content
                })
            else:
                formatted_results.append({
                    "title": f"Unexpected result type: {type(result)}",
                    "url": "",
                    "snippet": str(result)
                })
        
        logger.debug(f"Formatted results: {formatted_results}")
        return {"success": True, "results": formatted_results}
    except Exception as e:
        logger.error(f"Error performing Tavily search: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Error performing Tavily search: {str(e)}"}