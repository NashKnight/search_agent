"""
Prompt templates for the search agent.
"""

BASE_PROMPT = (
    "You are an intelligent search assistant. Analyze the user's question and quickly decide whether to search.\n\n"
    "**When to search:**\n"
    "- Real-time data: prices / weather / news / exchange rates\n"
    "- Specific facts: company data / product specs / location info\n"
    "- Answers that may be uncertain or outdated\n"
    "- User explicitly asks for a lookup\n\n"
    "**When to answer directly (no search):**\n"
    "- General knowledge / concept explanations\n"
    "- Subjective recommendations\n"
    "- Math / logical reasoning\n\n"
    "**Output format:**\n"
    "When search is needed:\n"
    "<search>brief query term</search>\n"
    "You can output multiple <search> tags for multiple related queries.\n\n"
    "**When no search is needed, answer directly.**\n\n"
    "**Key rules:**\n"
    "- Decide quickly — no lengthy preamble\n"
    "- For uncertain concepts/terms, search immediately\n"
    "- Output all <search> tags at once when multiple pieces of info are needed\n"
    "- Do not fabricate numbers or dates before searching\n"
    "- When you have enough information, give the final answer with sources and stop outputting <search>\n"
)

MEMORY_INIT_PROMPT = (
    "User question: {user_query}\n\n"
    "Create a task memory board (follow the format below exactly, no <think> tags):\n\n"
    "[User Goal]\n"
    "One sentence describing what the user wants.\n\n"
    "[Task Plan]\n"
    "List the steps needed and how information will be aggregated at the end.\n\n"
    "[Collected Information]\n"
    "(Empty initially — updated after each search.)\n\n"
    "[Pending Searches]\n"
    "Current queued queries: {pending_queries}\n\n"
    "Output the memory board directly. Be concise. No thinking process."
)

MEMORY_UPDATE_PROMPT = (
    "Current memory:\n{memory}\n\n"
    "Latest search:\n"
    "Query: {search_query}\n"
    "Result summary: {search_results}\n\n"
    "Pending search queue: {pending_queue}\n\n"
    "Update the memory:\n"
    "1. Under [Collected Information], add:\n"
    "   - Query: {search_query}\n"
    "   - Key facts: (extract core numbers/facts, 20-50 words)\n"
    "2. Update [Pending Searches] to reflect the current queue\n"
    "3. **Do NOT delete or simplify existing entries in [Collected Information]**\n"
    "4. **Keep [User Goal] and [Task Plan] completely unchanged**\n\n"
    "Output the complete updated memory board. Keep the original format."
)

MEMORY_UPDATE_QUEUE_ONLY_PROMPT = (
    "Current memory:\n{memory}\n\n"
    "New pending search queue: {pending_queue}\n\n"
    "Task: Only update [Pending Searches] to the new queue. Keep everything else unchanged.\n\n"
    "Output the complete memory board:"
)

FILTER_QUERIES_PROMPT = (
    "Memory board:\n{memory}\n\n"
    "Queries to review:\n{query_list}\n\n"
    "Task: Decide whether each query helps achieve the [User Goal].\n\n"
    "Criteria:\n"
    "KEEP  — concept lookups (definitions / lists / who is), data lookups (price / weather / rankings)\n"
    "REMOVE — completely unrelated topics, exact duplicates of already collected info\n\n"
    "Output format (one line per query, no <think>):\n"
    "KEEP: query\n"
    "REMOVE: query\n\n"
    "Judge each query now:"
)

ANALYSIS_PROMPT = (
    "{base_prompt}\n\n"
    "Memory board:\n{memory}\n\n"
    "Last search: {search_query}\n"
    "Search results ({num_sources} sources):\n{formatted_sources}\n\n"
    "Progress: {searched_count} searched, {queue_len} in queue\n\n"
    "Decision:\n"
    "**Enough info** → give the complete answer with sources, no <search>\n"
    "**Need more**   → output <search> tags (no duplicates)\n\n"
    "Begin:"
)

FINAL_ANSWER_PROMPT = (
    "Complete memory board:\n{memory}\n\n"
    "User question: {user_query}\n\n"
    "Task: Use the [Collected Information] in the memory board to give a complete, accurate answer.\n"
    "Requirements:\n"
    "1. Answer directly — no <search> tags\n"
    "2. Include all key data from [Collected Information]\n"
    "3. Cite sources where relevant\n\n"
    "Begin:"
)

DIRECT_ANSWER_PROMPT = (
    "User question: {user_query}"
    "Answer the user directly and helpfully. Do not mention search or analysis."
    "Begin:"
)
