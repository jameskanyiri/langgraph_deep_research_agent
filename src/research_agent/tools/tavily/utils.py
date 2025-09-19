import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Literal
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from src.research_agent.schema import Summary
from src.research_agent.tools.tavily.prompt import SUMMARIZE_WEBPAGE_CONTENT_PROMPT
from src.utils import get_today_str

load_dotenv(override=True)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm = init_chat_model(model="gpt-4o", temperature=0, timeout=60)

summarization_model = llm.with_structured_output(Summary)


## Multiple queries search using tavily client
def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[Dict]:
    """
    Perform search using tavily client api for multiple queries.

    Args:
        search_queries: List of search queries to perform
        max_results: Maximum number of results to return for each query
        topic: Topic of the search
        include_raw_content: Whether to include raw content in the results

    Returns:
        List of search results
    """
    results = []
    for query in search_queries:
        try:
            print(f"Searching for: {query}")
            result = tavily_client.search(
                query,
                max_results=max_results,
                topic=topic,
                include_raw_content=include_raw_content,
            )
            results.append(result)
            print(f"Found {len(result.get('results', []))} results for query: {query}")
        except Exception as e:
            print(f"Error searching for query '{query}': {str(e)}")
            # Add empty result to maintain structure
            results.append({"results": [], "query": query, "error": str(e)})
    return results


def deduplicate_search_results(search_results: List[Dict]) -> dict:
    """
    Deduplicate the search result by url to avoid processing duplicate content.

    Args:
        search_results: List of search results dictionaries

    Returns:
        Dictionary of deduplicated search results

    """

    unique_results = {}

    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = result

    return unique_results


def summarize_webpage_content(webpage_content: str) -> str:
    """
    Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw content of the webpage

    Returns:
        Summarized content of the webpage
    """
    try:
        print("Starting webpage content summarization...")

        # Format the system instruction with the current date and the webpage content
        system_instruction = SUMMARIZE_WEBPAGE_CONTENT_PROMPT.format(
            date=get_today_str(), webpage_content=webpage_content
        )

        # Prepare the messages for the summarization model
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": "Summarize the following webpage content"},
        ]

        # Generate the summary with timeout
        print("Calling summarization model...")
        summary = summarization_model.invoke(messages)

        # Format summary with clear structure
        formatted_summary = (
            f"<summary> \n{summary.summary}\n </summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n </key_excerpts>\n\n"
        )

        print("Summarization completed successfully")
        return formatted_summary

    except asyncio.CancelledError:
        print("Summarization was cancelled")
        return "Summarization was cancelled due to timeout or interruption."
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return f"Error summarizing content: {str(e)}"


def process_search_results(unique_results: Dict) -> Dict:
    """
    Process the search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """

    summarized_results = {}

    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result["content"]
        else:
            # Summarize raw content for better processing
            content = summarize_webpage_content(result["raw_content"])

        summarized_results[url] = {
            "title": result.get("title", ""),
            "content": content,
        }
    return summarized_results


def format_search_results(summarized_results: Dict) -> str:
    """
    Format the summarized search results into a well structured string output.

    Args:
        summarized_results: Dictionary of summarized search results

    Returns:
        Well structured string output of the summarized search results
    """

    if not summarized_results:
        return "No search results found. Please try a different search query."

    formatted_results = " Search results:\n\n "

    # Format the results into a well structured string output
    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_results += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_results += f"URL: {url}\n\n"
        formatted_results += f"SUMMARY:\n{result['content']}\n\n"
        formatted_results += "-" * 100 + "\n"

    return formatted_results
