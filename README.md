# Web Scraper Agent with AI Processing

An intelligent web scraper agent that combines web scraping capabilities with AI-powered content analysis. This agent can scrape websites, process their content, and provide meaningful insights through multiple processing stages.

## 🌟 Features

- **Web Scraping**: Automatically scrapes and cleans content from any provided URL
- **Content Classification**: Categorizes content into predefined categories
- **Content Summarization**: Generates concise 2-3 sentence summaries
- **Tag Extraction**: Identifies 5-7 most relevant tags
- **Related Topics**: Suggests 3-5 related topics for further research
- **Sentiment Analysis**: Analyzes emotional tone with detailed scoring
- **Key Phrase Extraction**: Identifies important quotes and phrases with context
- **Readability Analysis**: Evaluates content complexity and target audience
- **Fact Checking**: Identifies claims needing verification
- **Content Structure Analysis**: Analyzes organization and flow

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/MegumiinUwU/AI-Web-Scraper-Agent.git
cd AI-Web-Scraper-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

4. Run


## 📚 Documentation

For detailed information about:
- Code structure and implementation
- Processing nodes and their functions
- Workflow setup and configuration
- Best practices and use cases
- Future improvements and contributions

Please refer to [LEARN.md](LEARN.md) in this repository.

## 🛠️ Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - langchain
  - langchain-groq
  - langgraph
  - beautifulsoup4
  - requests
  - python-dotenv



## 📝 License

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
```

## ⚠️ Important Note

This project requires a Groq API key for the AI processing features. Make sure to:
1. Sign up for a Groq account
2. Obtain your API key
3. Add it to your `.env` file
