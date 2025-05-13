import os
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import re
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Initialize the ChatGroq instance
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)

# Define the state structure
class State(TypedDict):
    url: str  # The URL to scrape
    scraped_content: str  # The content scraped from the website
    classification: str  # Classification of the content
    summary: str  # Summary of the content
    tags: List[str]  # Popular tags extracted from the content
    related_topics: List[str]  # Suggested related topics
    sentiment: str  # Sentiment analysis
    key_phrases: List[str]  # Key phrases and quotes
    readability: str  # Readability analysis
    facts_to_verify: List[str]  # Facts that need verification
    structure: str  # Content structure analysis

def scrape_website(url: str) -> str:
    """
    Scrape content from a website and save it to a file.
    """
    try:
        # Get the domain name for the file
        domain = urlparse(url).netloc
        filename = f"{domain}.txt"
        
        # Fetch the webpage
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return text
    
    except Exception as e:
        print(f"Error scraping website: {str(e)}")
        return ""

def classification_node(state: State):
    """
    Classify the content into predefined categories.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Analyze the following content and classify it into one of these categories: 
        Technology, Business, Science, Health, Entertainment, Education, or Other.
        
        Content: {content}
        
        Category:"""
    )

    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

def summarize_node(state: State):
    """
    Create a summary of the content.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Provide a concise summary of the following content in 2-3 sentences.
        
        Content: {content}
        
        Summary:"""
    )
    
    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

def extract_tags_node(state: State):
    """
    Extract popular tags from the content.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Analyze the following content and extract 5-7 most relevant tags that represent the main topics.
        Return the tags as a comma-separated list.
        
        Content: {content}
        
        Tags:"""
    )
    
    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    tags = llm.invoke([message]).content.strip().split(", ")
    return {"tags": tags}

def suggest_topics_node(state: State):
    """
    Suggest related topics for further research.
    """
    prompt = PromptTemplate(
        input_variables=["content", "tags"],
        template="""Based on the following content and its tags, suggest 3-5 related topics that would be interesting to explore further.
        Return the topics as a comma-separated list.
        
        Content: {content}
        Tags: {tags}
        
        Related Topics:"""
    )
    
    message = HumanMessage(content=prompt.format(
        content=state["scraped_content"],
        tags=", ".join(state["tags"])
    ))
    topics = llm.invoke([message]).content.strip().split(", ")
    return {"related_topics": topics}

def sentiment_analysis_node(state: State):
    """
    Analyze the sentiment of the content.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Analyze the sentiment of the following content. 
        Provide a sentiment score from -1 (very negative) to 1 (very positive) and a brief explanation.
        Format: Score: [number], Explanation: [text]
        
        Content: {content}
        
        Sentiment Analysis:"""
    )
    
    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    sentiment = llm.invoke([message]).content.strip()
    return {"sentiment": sentiment}

def key_phrases_node(state: State):
    """
    Extract key phrases and important quotes from the content.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Extract 3-5 key phrases or important quotes from the following content.
        For each phrase/quote, provide a brief context of why it's important.
        Format each entry as: "Phrase: [text] - Context: [explanation]"
        
        Content: {content}
        
        Key Phrases:"""
    )
    
    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    key_phrases = llm.invoke([message]).content.strip().split("\n")
    return {"key_phrases": key_phrases}

def readability_score_node(state: State):
    """
    Calculate the readability score and suggest the target audience.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Analyze the readability of the following content.
        Provide:
        1. A readability score (1-10, where 10 is most complex)
        2. The suggested target audience (e.g., "General Public", "Academic", "Technical")
        3. Brief explanation of the complexity level
        
        Content: {content}
        
        Readability Analysis:"""
    )
    
    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    readability = llm.invoke([message]).content.strip()
    return {"readability": readability}

def fact_check_node(state: State):
    """
    Identify potential facts and claims that need verification.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Identify 3-5 key facts or claims from the following content that might need verification.
        For each fact/claim, provide:
        1. The statement
        2. Why it might need verification
        3. Suggested sources to verify
        
        Content: {content}
        
        Facts to Verify:"""
    )
    
    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    facts_to_verify = llm.invoke([message]).content.strip().split("\n")
    return {"facts_to_verify": facts_to_verify}

def content_structure_node(state: State):
    """
    Analyze the structure and organization of the content.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Analyze the structure and organization of the following content.
        Provide:
        1. The main sections/topics
        2. The logical flow of the content
        3. Suggestions for better organization (if any)
        
        Content: {content}
        
        Structure Analysis:"""
    )
    
    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    structure = llm.invoke([message]).content.strip()
    return {"structure": structure}

# Create and configure the workflow
workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classify_content", classification_node)
workflow.add_node("summarize_content", summarize_node)
workflow.add_node("extract_content_tags", extract_tags_node)
workflow.add_node("suggest_related_topics", suggest_topics_node)
workflow.add_node("analyze_sentiment", sentiment_analysis_node)
workflow.add_node("extract_key_phrases", key_phrases_node)
workflow.add_node("analyze_readability", readability_score_node)
workflow.add_node("check_facts", fact_check_node)
workflow.add_node("analyze_structure", content_structure_node)

# Add edges to the graph
workflow.set_entry_point("classify_content")
workflow.add_edge("classify_content", "summarize_content")
workflow.add_edge("summarize_content", "extract_content_tags")
workflow.add_edge("extract_content_tags", "suggest_related_topics")
workflow.add_edge("suggest_related_topics", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "extract_key_phrases")
workflow.add_edge("extract_key_phrases", "analyze_readability")
workflow.add_edge("analyze_readability", "check_facts")
workflow.add_edge("check_facts", "analyze_structure")
workflow.add_edge("analyze_structure", END)

# Compile the graph
app = workflow.compile()

# Test the agent
if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/JoJo%27s_Bizarre_Adventure"  # Replace with your test URL
    scraped_content = scrape_website(test_url)
    
    if scraped_content:
        result = app.invoke({
            "url": test_url,
            "scraped_content": scraped_content
        })
        
        print("\nClassification:", result["classification"])
        print("\nSummary:", result["summary"])
        print("\nTags:", result["tags"])
        print("\nRelated Topics:", result["related_topics"])
        print("\nSentiment Analysis:", result["sentiment"])
        print("\nKey Phrases:")
        for phrase in result["key_phrases"]:
            print(f"- {phrase}")
        print("\nReadability Analysis:", result["readability"])
        print("\nFacts to Verify:")
        for fact in result["facts_to_verify"]:
            print(f"- {fact}")
        print("\nContent Structure:", result["structure"]) 