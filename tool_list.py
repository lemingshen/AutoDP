from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def sanitize_code(code: str):
    start_index = code.find("```python")
    end_index = code.rfind("```")

    return code[start_index + 9 : end_index].strip()


model = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
web_search_tool = TavilySearchResults(
    max_results=10,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)
web_loader = WebBaseLoader(encoding="utf-8")
agent_memory = MemorySaver()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
