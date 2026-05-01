
import re
import time
import requests
import trafilatura
from bs4 import BeautifulSoup
from readability import Document
import pandas as pd
import json
import re
import requests
from bs4 import BeautifulSoup
system_instructions = r"""<<BEGIN-CODEBOOK-RULES>>
You are a strict annotation engine. Follow this codebook exactly. Base decisions only on the provided message text and artifacts. 
When making your decision, please use the provided information sources in the following order of priority:

# Core Principles
1.  **Decision Order**: Label strictly in this order: Q1 (Theme) → Q2 (Claim type) → Q3 (CTAs) → Q4 (Evidence).
2.  **51% Threshold**: Apply a label only if it's more probable than not (≥51%) to be correct.
3.  **Theme (Q1)**: Single-label, unless two themes both pass 51% and each covers ≥35% of the content.
4.  **CTAs (Q3)**: Multi-label. Apply all that meet the 51% threshold.
5.  **Tie-breakers**: If tied, use this order: token share → domain primacy → first-mention.

# Q1: Theme (Primary topic of the message)
-   **Finance/Crypto**: Financial assets, markets, trading, crypto/tokens.
-   **Public health & medicine**: Disease, vaccines, clinical guidance, health policy.
-   **Politics**: Partisan opinion, elections, government administration/policy.
-   **Crime & public safety**: Crimes, threats, policing, safety alerts.
-   **News/Information**: General updates when no other specific domain fits. (Use as a last resort).
-   **Technology**: R&D, engineering, software/product features.
-   **Lifestyle & well‑being**: Non-clinical personal improvement (fitness, diet, productivity).
-   **Gaming/Gambling**: Games of chance/skill, wagering, betting, casinos, slots.
-   **Sports**: Athletic competitions, teams, fixtures, results.
-   **Conversation/Chat/Other**: Greetings, housekeeping, channel meta, or unclassifiable chat.
-   **Other (Theme)**: Extremely rare; use only when no other label fits.

# Q2: Claim type / Quality (Characterize the message's proposition)
(Multi-label, max 3. Follow precedence and forbidden pairs.)

**Precedence (First-match-wins for PRIMARY label):**
1.  **No substantive claim**: No checkable assertion (greetings, housekeeping, bare URL). Past-tense verbs with outcomes are substantive.
2.  **Announcement**: Schedule, availability, or housekeeping only. No performance metrics.
3.  **Speculative forecast / prediction**: Forward-looking claims (targets, trade calls, projections).
4.  **Promotional hype / exaggerated profit guarantee**: "guaranteed", "no risk", "5x/10x", "set to explode".
5.  **Scarcity/FOMO tactic**: Urgency/limited window ("last chance", "ends today", "only N left").
6.  **Misleading context / cherry‑picking**: Technically true but isolated wins without denominators/context.
7.  **Emotional appeal / fear‑mongering**: Primary purpose is to invoke fear/anger.
8.  **Rumour / unverified report**: Unnamed sources or rumour cues ("sources say", "allegedly", "leaked").
9.  **Opinion / subjective statement**: Normative judgments (best/worst, should/unfair, "I think").
10. **Verifiable factual statement**: A checkable present/past fact with a named entity/action.
11. **Other (Claim type)**: Extremely rare.

**Forbidden Pairs**:
-   (Rumour / unverified report + Verifiable factual statement)
-   (Announcement + Verifiable factual statement)
-   (No substantive claim + any other)

# Q3: Call-to-Action (What the audience is asked to do)
(Multi-label. If any CTA is present, do not use No CTA.)

-   **Share / repost / like**: Explicit requests to share, RT, like.
-   **Engage/Ask questions**: Asks for replies, comments, votes, feedback, DMs.
-   **Visit external link / watch video**: Assign if (a) explicit invite ("watch", "click", "read more", "link in bio", 👇/👉/➡️ + link) or (b) any substantive content and a URL appear together.
-   **Buy / invest / donate**: Spending/trading directives (buy/sell/hold/long/short) or explicit trade setups (Entry + TP or SL). **Guard**: Do not assign for past-tense recaps ("TP hit", "profit +28%") with no forward directive.
-   **Join/Subscribe**: "join", "subscribe", "follow", "register", "whitelist".
-   **Attend event / livestream**: Broadcast language ("live now", "stream", "AMA", "webinar") or a specific time/date for an event.
-   **No CTA**: Applies only when none of the above are triggered. A bare URL is No CTA.

# Q4: Evidence / Support Shown (Form of support included)
(Multi-label.)

-   **None / assertion only**: A claim with zero attached support.
-   **Link/URL**: Any external URL, shortlink, or off-platform pointer.
-   **Quotes/Testimony**: Direct quote or statement attributed to a named person/organization.
-   **Statistics**: Numeric evidence with measure/scope (counts, %, prices, durations). Not dates/version numbers.
-   **Chart / price graph / TA diagram**: Any plotted visual or a link to a charting host (e.g., TradingView).
-   **Other (Evidence)**: Support that doesn't fit above (e.g., on-chain hashes, audit badge screenshots without links).



# STRICT OUTPUT SCHEMA (MUST FOLLOW)
{
"theme": "...",
"claim_types": "...",
"ctas": "...",
"evidence": "..."
}
<<END-CODEBOOK-RULES>>"""
def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Performs a DuckDuckGo search. Returns a list of {title, url, snippet}.
    """
    # 在函数内部导入，以避免在不需要时报错
    from ddgs import DDGS
    
    
    # print(f"Performing web search for query: '{query}'...")
    results = []
    # try:
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
            for r in search_results:
                results.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body"),
                })
    except :
        # 这个 except 块会专门捕捉“无结果”的异常
        # print(f"!!! Web search for '{query}' returned no results or failed. Skipping search for this item.")
        # 返回一个包含错误信息的列表，或者直接返回一个空列表
        return [{"error": f"No results found for query: {query}"}]
    
    return results



def _extract_with_multiple_tools(html_content: bytes):
    """尝试用多种工具提取内容"""
    
    # 1. 首选 Trafilatura (您已在使用)
    content = trafilatura.extract(html_content)
    if content:
        # 基本的清洗，去除过多空行
        return '\n'.join(line for line in content.splitlines() if line.strip())

    # 2. 备选 Readability
    try:
        soup = BeautifulSoup(html_content, 'lxml') 
    except Exception:
        # 如果 lxml 解析失败，则回退到 Python 内置的解析器
        
        soup = BeautifulSoup(html_content, 'html.parser')
    doc = Document(str(soup))
    # 提取HTML格式的主要内容，然后转为纯文本
    html_summary = doc.summary()
    soup = BeautifulSoup(html_summary, 'html.parser')
    content = soup.get_text(separator='\n', strip=True)
    if content:
        return content
def _fetch_with_requests(url: str):
    """Helper function: Tries to fetch content using the fast 'requests' library."""
    try:
        print(f"-> Attempting FAST fetch with 'requests' for: {url}")
        headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'cache-control': 'max-age=0',
    'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}
        response = requests.get(url, headers=headers, timeout=10,verify=False)
        response.raise_for_status()
        # print(response.text)
        main_content = _extract_with_multiple_tools(response.content)
        
        # **关键检查**：检查返回的是否是"请开启JS"之类的无用内容
        if main_content and "javascript" in main_content.lower() and "disabled" in main_content.lower():
            # print("   - 'requests' fetch successful, but content is a JavaScript gate. Needs Selenium.")
            return None # 返回None，表示需要升级到Selenium
            
        if main_content:
            # print("   - SUCCESS: 'requests' fetch was successful.")
            return main_content
        else:
            # print("   - FAILED: 'requests' worked but Trafilatura found no content.")
            return None # Trafilatura 提取失败，也尝试Selenium

    except requests.exceptions.RequestException as e:
        # print(f"   - FAILED: 'requests' failed with error: {e}. Will try Selenium.")
        return None # 任何requests错误都意味着需要尝试Selenium





def extract_and_append_web_content(text: str) -> str:
    """
    Main function using a hybrid strategy.
    First, it tries a fast 'requests' fetch. If that fails or returns a
    JavaScript-gate, it automatically falls back to a full 'Selenium' render.
    """
    # 步骤 1: URL 清理 (逻辑保持不变)
    rough_url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(rough_url_pattern, text)
    if not match: return text
    candidate_url = match.group(0)
    clean_url_pattern = r'^(https?:\/\/[a-zA-Z0-9\/:._?=&%#~+-@]+)'
    clean_match = re.search(clean_url_pattern, candidate_url)
    if not clean_match: return text
    url = clean_match.group(0)

    # 步骤 2: 混合策略获取内容
    main_content = None
    
    # 优先处理X.com/Twitter，因为我们100%确定它需要Selenium
    
    main_content = _fetch_with_requests(url)
        # 如果requests失败(返回None)，则自动切换到Selenium

    # 步骤 3: 附加内容并返回
    if main_content:
        MAX_LENGTH = 4000
        if len(main_content) > MAX_LENGTH:
            main_content = main_content[:MAX_LENGTH] + " [Content Truncated]"
        return f"{text}\n\n--- Appended Web Content ---\n{main_content}"
    else:
        # print(f"!!! All fetch methods failed for URL '{url}'. Skipping content append.")
        return text

def process_row_message(message_text):
        # --- 步骤 1: 如果存在URL，先提取其内容 ---
        # 无论如何，我们都将进行搜索，所以先处理URL
        if "http" in message_text:
            processed_text = extract_and_append_web_content(message_text)
        # processed_text =message_text
            if processed_text==message_text :
                if len(message_text)<50:
            # --- 步骤 2: 对原始消息文本执行网页搜索 ---
            # 我们使用原始的、更简洁的 message_text 进行搜索，以获得更相关的结果
                    max_results=3
                elif len(message_text)<100:
                    max_results=2
                else:
                    max_results=0
                search_results = web_search(message_text, max_results=max_results) # 可在此处调整数量
       
            
        # max_results=0
        

        # --- 步骤 3: 格式化搜索结果并附加 ---
                if search_results and not search_results[0].get("error") and max_results!=0:
                    formatted_snippets = []
                    # print(len(search_results))
                    for i, res in enumerate(search_results[:max_results]):
                        title = res.get('title', 'No Title')
                        snippet = res.get('snippet', 'No Snippet')
                        url = res.get('url', 'No URL')
                        formatted_snippets.append(f"Source [{i+1}]: {title}\nSnippet: {snippet}\nURL: {url}\n")
            
            # 将搜索结果附加到已处理的文本后面
                    search_context = "\n\n--- Appended Web Search Results ---\n" + "\n".join(formatted_snippets)
                    processed_text += search_context
        else:
            processed_text=message_text
        return processed_text

def get_sample(original_message):
    processed_message = process_row_message(original_message)
    message_list = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": f"message text:{processed_message}"},
        ]
    return message_list