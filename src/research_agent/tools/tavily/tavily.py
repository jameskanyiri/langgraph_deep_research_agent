import os
from dotenv import load_dotenv
from typing import Annotated, List, Dict, Literal
from src.research_agent.tools.tavily.utils import (
    tavily_search_multiple,
    deduplicate_search_results,
    process_search_results,
    format_search_results,
)
from langchain_core.tools import tool, InjectedToolArg

load_dotenv(override=True)


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """
    Perform search using tavily client api for a single query.

    Args:
        query: A Single search query to execute
        max_results: Maximum number of results to return
        topic: Topic of to filter results by ("general", "news", "finance")

    Returns:
        Formatted string of search results with summaries
    """
    try:
        print(f"Starting Tavily search for query: '{query}'")

        # Execute search for a single query
        search_result = tavily_search_multiple(
            [query], max_results=max_results, topic=topic, include_raw_content=True
        )

        # Deduplicate result by url to avoid processing duplicate context
        unique_results = deduplicate_search_results(search_result)
        print(f"Found {len(unique_results)} unique results")

        # Process the results for summarization
        summarized_results = process_search_results(unique_results)

        # Format output for consumption by the research agent
        formatted_output = format_search_results(summarized_results)

        print("Tavily search completed successfully")
        return formatted_output

    except Exception as e:
        error_msg = f"Error during Tavily search for query '{query}': {str(e)}"
        print(error_msg)
        return error_msg
