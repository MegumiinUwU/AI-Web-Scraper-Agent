# Web Scraper Agent with AI Processing

This project implements an intelligent web scraper agent that combines web scraping capabilities with AI-powered content analysis. The agent can scrape websites, process their content, and provide meaningful insights through multiple processing stages.

## Code Structure and Explanation

### 1. Imports and Setup
```python
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
```
This section:
- Imports necessary libraries for web scraping, AI processing, and type hints
- Loads environment variables from .env file
- Initializes the Groq LLM with specific model and temperature settings

### 2. State Definition
```python
class State(TypedDict):
    url: str  # The URL to scrape
    scraped_content: str  # The content scraped from the website
    classification: str  # Classification of the content
    summary: str  # Summary of the content
    tags: List[str]  # Popular tags extracted from the content
    related_topics: List[str]  # Suggested related topics
```
This defines the data structure that will be passed between nodes in our workflow.

### 3. Web Scraping Function
```python
def scrape_website(url: str) -> str:
    try:
        # Get the domain name for the file
        domain = urlparse(url).netloc
        filename = f"{domain}.txt"
        
        # Fetch the webpage with headers to mimic a browser
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content and clean it
        text = soup.get_text()
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
```
This function:
- Takes a URL as input
- Extracts the domain name for the output file
- Fetches the webpage with proper headers
- Parses and cleans the HTML content
- Removes unnecessary elements (scripts, styles)
- Saves the cleaned content to a file
- Returns the cleaned text or empty string if there's an error

### 4. Processing Nodes

#### Classification Node
```python
def classification_node(state: State):
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
```
This node:
- Takes the scraped content
- Uses a prompt template to ask the AI to classify the content
- Returns the classification result

#### Summarization Node
```python
def summarize_node(state: State):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Provide a concise summary of the following content in 2-3 sentences.
        
        Content: {content}
        
        Summary:"""
    )
    
    message = HumanMessage(content=prompt.format(content=state["scraped_content"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}
```
This node:
- Takes the scraped content
- Uses a prompt to generate a concise summary
- Returns the summary

#### Tag Extraction Node
```python
def extract_tags_node(state: State):
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
```
This node:
- Takes the scraped content
- Uses a prompt to extract relevant tags
- Returns a list of tags

#### Topic Suggestion Node
```python
def suggest_topics_node(state: State):
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
```
This node:
- Takes both the content and previously extracted tags
- Uses a prompt to suggest related topics
- Returns a list of related topics

#### Sentiment Analysis Node
```python
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
```
This node:
- Takes the scraped content
- Uses a prompt to analyze the emotional tone
- Returns a sentiment score and explanation
- Helps understand the overall tone of the content

#### Key Phrases Node
```python
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
```
This node:
- Takes the scraped content
- Identifies the most significant phrases or quotes
- Provides context for each important element
- Returns a list of key phrases with explanations

#### Readability Score Node
```python
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
```
This node:
- Takes the scraped content
- Evaluates the complexity level
- Determines the appropriate audience
- Returns a comprehensive readability analysis

#### Fact Check Node
```python
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
```
This node:
- Takes the scraped content
- Identifies claims that need verification
- Provides reasoning for verification
- Suggests reliable sources for fact-checking

#### Content Structure Node
```python
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
```
This node:
- Takes the scraped content
- Analyzes the organization and flow
- Identifies main sections
- Suggests structural improvements

### 5. Workflow Setup
```python
# Create and configure the workflow
workflow = StateGraph(State)

# Add all nodes to the graph
workflow.add_node("classify_content", classification_node)
workflow.add_node("summarize_content", summarize_node)
workflow.add_node("extract_content_tags", extract_tags_node)
workflow.add_node("suggest_related_topics", suggest_topics_node)
workflow.add_node("analyze_sentiment", sentiment_analysis_node)
workflow.add_node("extract_key_phrases", key_phrases_node)
workflow.add_node("analyze_readability", readability_score_node)
workflow.add_node("check_facts", fact_check_node)
workflow.add_node("analyze_structure", content_structure_node)

# Add edges to the graph with a more complex flow
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
```
This section:
- Creates a state graph for the workflow
- Adds all processing nodes including the new ones
- Sets up a more comprehensive sequence of operations
- Compiles the workflow into an executable application

### 6. Main Execution
```python
if __name__ == "__main__":
    test_url = "https://example.com"  # Replace with your test URL
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
```
This section:
- Defines the main execution flow
- Scrapes the website
- Processes the content through the enhanced AI pipeline
- Prints all analysis results including the new features

## Features

1. **Web Scraping**
   - Automatically scrapes content from any provided URL
   - Cleans and formats the scraped content
   - Saves the content to a text file named after the website's domain
   - Handles errors gracefully

2. **Content Classification**
   - Categorizes content into predefined categories:
     - Technology
     - Business
     - Science
     - Health
     - Entertainment
     - Education
     - Other

3. **Content Summarization**
   - Generates concise 2-3 sentence summaries
   - Captures the main points of the content

4. **Tag Extraction**
   - Identifies 5-7 most relevant tags
   - Helps in content categorization and searchability

5. **Related Topics Suggestion**
   - Suggests 3-5 related topics for further research
   - Uses both content and extracted tags for suggestions

6. **Sentiment Analysis**
   - Analyzes the emotional tone of the content
   - Provides a sentiment score from -1 (very negative) to 1 (very positive)
   - Includes detailed explanation of the sentiment

7. **Key Phrase Extraction**
   - Identifies important quotes and phrases
   - Provides context for each significant phrase
   - Highlights the most impactful parts of the content

8. **Readability Analysis**
   - Calculates content complexity on a 1-10 scale
   - Determines the target audience
   - Provides complexity level explanation
   - Helps in content accessibility assessment

9. **Fact Checking Assistance**
   - Identifies claims that need verification
   - Explains why verification is necessary
   - Suggests reliable sources for fact-checking
   - Helps maintain content credibility

10. **Content Structure Analysis**
    - Analyzes the organization of the content
    - Identifies main sections and topics
    - Evaluates the logical flow
    - Suggests structural improvements

## How It Works

### 1. Setup and Dependencies

First, install the required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- langchain
- langchain-groq
- langgraph
- beautifulsoup4
- requests
- python-dotenv

### 2. Environment Setup

Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

### 3. Core Components

#### Web Scraping Function
```python
def scrape_website(url: str) -> str:
    # Scrapes website content
    # Cleans and formats the text
    # Saves to a file named after the domain
    # Returns the cleaned content
```

#### Processing Nodes
1. **Classification Node**
   - Analyzes content and assigns a category
   - Uses AI to understand the main theme

2. **Summarization Node**
   - Creates concise summaries
   - Focuses on key information

3. **Tag Extraction Node**
   - Identifies main topics
   - Creates relevant tags

4. **Topic Suggestion Node**
   - Suggests related topics
   - Uses content and tags for context

### 4. Workflow

The agent uses a sequential workflow:
1. Scrape website content
2. Classify the content
3. Generate summary
4. Extract tags
5. Suggest related topics

## Usage Example

```python
from web_agent import app, scrape_website

# Provide the URL to scrape
url = "https://example.com"
scraped_content = scrape_website(url)

if scraped_content:
    # Process the content through the AI pipeline
    result = app.invoke({
        "url": url,
        "scraped_content": scraped_content
    })
    
    # Access the results
    print("Classification:", result["classification"])
    print("Summary:", result["summary"])
    print("Tags:", result["tags"])
    print("Related Topics:", result["related_topics"])
```

## Output Format

The agent provides structured output:
1. **Classification**: Single category label
2. **Summary**: 2-3 sentence summary
3. **Tags**: List of 5-7 relevant tags
4. **Related Topics**: List of 3-5 suggested topics

## Use Cases

1. **Content Analysis**
   - Quickly understand the main themes of web content
   - Generate metadata for content management systems

2. **Research Assistance**
   - Identify related topics for further research
   - Get quick summaries of long articles

3. **Content Categorization**
   - Automatically categorize web content
   - Generate relevant tags for content organization

4. **Information Extraction**
   - Extract key information from websites
   - Generate structured data from unstructured web content

## Best Practices

1. **Error Handling**
   - Always check if scraping was successful
   - Handle network errors gracefully

2. **Rate Limiting**
   - Be respectful of website resources
   - Implement delays between requests if scraping multiple pages

3. **Content Processing**
   - Clean and format content before AI processing
   - Remove unnecessary elements (scripts, styles)

4. **API Usage**
   - Keep API keys secure
   - Monitor API usage and costs

## Future Improvements

1. **Enhanced Scraping**
   - Add support for dynamic content
   - Implement JavaScript rendering

2. **Advanced Processing**
   - Add sentiment analysis
   - Implement entity recognition

3. **Output Formats**
   - Add support for different output formats (JSON, XML)
   - Implement custom output templates

4. **Performance**
   - Add caching mechanisms
   - Implement parallel processing

## Contributing

Feel free to contribute to this project by:
1. Reporting bugs
2. Suggesting new features
3. Improving documentation
4. Adding new processing nodes

## License

This project is open source and available under the MIT License.

## Example Output

Here's an example of the web agent's output when analyzing a Wikipedia page about JoJo's Bizarre Adventure:

```
Classification: Entertainment.

Summary: Here is a 2-3 sentence summary of JoJo's Bizarre Adventure:

JoJo's Bizarre Adventure is a Japanese manga series written and illustrated by Hirohiko Araki, divided into nine main story arcs, each following a new protagonist bearing the "JoJo" nickname. The series is known for its unique art style, poses, and references to Western popular music and fashion, and has spawned a media franchise including manga, anime, video games, and a live-action film. With over 120 million copies in circulation, it is one of the best-selling manga series in history, and has received critical acclaim for its storytelling, characters, and artwork.

Tags: ['manga', 'anime', 'japanese media', 'adventure', 'supernatural', 'action', 'fantasy', 'shonen', 'seinen', 'gucci']

Related Topics: ['Japanese manga and anime series', 'Gucci collaborations', 'Fantasy adventure stories', 'Shonen and seinen manga', 'Supernatural action series.']

Sentiment Analysis: Score: 0.9
Explanation: The sentiment of the JoJo's Bizarre Adventure Wikipedia page is overwhelmingly positive. The page presents a comprehensive overview of the series, highlighting its achievements, critical acclaim, and lasting impact on the world of manga and anime. The text emphasizes the series' unique art style, engaging storyline, and memorable characters, which have contributed to its massive popularity and influence. While the page also mentions some controversy and criticism, the overall tone is celebratory and enthusiastic, reflecting the series' dedicated fan base and its status as a cultural phenomenon.

Key Phrases:
- Here are 5 key phrases or important quotes from the content, along with a brief context for each:
-
- * "Mystery is the central theme of the manga" - Context: This quote from Hirohiko Araki highlights the importance of mystery in JoJo's Bizarre Adventure, which is a driving force behind the series' intricate plot and character developments.
- * "I wanted to try a different type of main character for every part" - Context: Araki's statement emphasizes his approach to creating diverse protagonists for each part of the series, which allows for a fresh and dynamic storytelling experience.
- * "JoJo-dachi" (ジョジョ立ち, lit. "JoJo standing") - Context: This term refers to the iconic poses adopted by characters in the series, which have become a signature element of JoJo's Bizarre Adventure's visual style and are inspired by Araki's studies of Michelangelo's sculptures.
- * "The supernatural basis of the fights in my series evened the battlefield for women and children to match up against strong men" - Context: Araki's comment highlights the series' use of supernatural elements to create a more level playing field for characters of different ages, sizes, and abilities, making the story more inclusive and exciting.
- * "Over 120 million copies in circulation" - Context: This impressive circulation figure makes JoJo's Bizarre Adventure one of the best-selling manga series in history, demonstrating its enduring popularity and influence across the globe.

Readability Analysis: **Readability Score: 8/10**

The content of the JoJo's Bizarre Adventure Wikipedia page has a relatively high readability score, indicating that it is moderately complex. The text assumes a certain level of prior knowledge about the series, manga, and anime, but it is still accessible to a general audience interested in learning more about the topic.

**Suggested Target Audience:**
* **Fans of the series:** The content is well-suited for fans of the JoJo's Bizarre Adventure series, who will appreciate the detailed information about the manga, anime, and related media.
* **Anime and manga enthusiasts:** The page is also suitable for enthusiasts of anime and manga in general, who will find the information about the series' history, production, and reception interesting.
* **General audience with some background knowledge:** While the content is specialized, it is still accessible to a general audience with some background knowledge about manga, anime, or Japanese culture.

**Complexity Level:**
The complexity level of the content is moderate, with:
* **Technical terms:** The page uses specialized terms related to manga, anime, and Japanese culture, which may require some explanation for readers without prior knowledge.
* **Detailed plot summaries:** The content includes detailed summaries of each part of the series, which may be dense for readers who are not familiar with the story.
* **References and citations:** The page includes numerous references and citations, which add to the complexity of the text.

Facts to Verify:
- Here are 3-5 key facts or claims from the content that might need verification:
-
- 1. **The number of copies of JoJo's Bizarre Adventure in circulation**: The article claims that JoJo's Bizarre Adventure had over 120 million copies in circulation by August 2023. This information might need verification from a reliable source, such as Shueisha or a reputable manga tracking website.
-
- 2. **The inspiration behind Araki's art style**: Araki mentions that he was inspired by Western art, such as a piece by Paul Gauguin, to use unusual colors in his art. While this might be true, verifying this information through an interview or a primary source could confirm its accuracy.
-
- 3. **The number of volumes published**: The article states that JoJo's Bizarre Adventure is the largest ongoing manga series published by Shueisha by number of volumes, with its chapters collected in 136 tankōbon volumes as of December 2024. Verifying this information through Shueisha's official publications or a reliable manga database could confirm its accuracy.
-
- 4. **The live-action film's release and production**: The article mentions that a live-action film based on Diamond Is Unbreakable was directed by Takashi Miike and released in Japan in 2017. Verifying this information through a reliable source, such as the film's official website or a reputable entertainment news outlet, could confirm its accuracy.
-
- 5. **The anime adaptation's episode count**: The article claims that David Production has produced five seasons of the anime series, consisting of 190 total episodes, adapting through the manga's sixth part, Stone Ocean. Verifying this information through the anime's official website or a reliable anime database could confirm its accuracy.

Content Structure: ## Analysis of JoJo's Bizarre Adventure Content Structure and Organization

### 1. Main Sections/Topics

The JoJo's Bizarre Adventure Wikipedia page is divided into several main sections/topics:

1. **Introduction**: A brief overview of the manga series, including its genre, publication history, and basic premise.
2. **Plot**: A detailed summary of the manga's story arcs, divided into parts (Phantom Blood to The JoJoLands).
3. **Production**: Insights into the creation of the manga, including inspiration, character design, and themes.
4. **Media**: A comprehensive overview of the various media formats in which JoJo's Bizarre Adventure has been released, including:
        * **Manga**: Publication history, volumes, and spin-offs.
        * **Anime**: Adaptation history, studios, and seasons.
        * **Other media**: Drama CDs, video games, light novels, and art books.
5. **Reception**: A discussion of the series' critical and commercial reception, including:
        * **Sales**: Circulation numbers and sales data.
        * **Critical reception**: Reviews and ratings from various sources.
        * **Accolades**: Awards and recognition received by the series.
6. **Legacy and collaborations**: The series' impact on popular culture, collaborations with other artists and brands (e.g., Gucci), and notable anniversaries.

### 2. Logical Flow of Content

The content flows logically from an introduction to the series, followed by a detailed exploration of its plot, production, media, reception, and legacy. The structure allows readers to navigate through the various aspects of JoJo's Bizarre Adventure, from its creation to its impact on popular culture.

### 3. Suggestions for Better Organization

While the current structure is well-organized, some potential improvements could be:

* **Consolidate similar sections**: Merge the "Production" and "Media" sections, as they both deal with the creation and adaptation of the series.
* **Expand the "Legacy" section**: Consider adding more information on the series' influence on other creators, its presence in popular culture, and fan engagement.
* **Use clearer headings and subheadings**: Implement a more consistent and descriptive heading structure to facilitate navigation and understanding.

Overall, the JoJo's Bizarre Adventure Wikipedia page provides a comprehensive and well-structured overview of the series, making it a valuable resource for fans and researchers alike. 