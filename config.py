import os

os.environ["OPENAI_API_KEY"] = "<Your OpenAI API Key>"
os.environ["TAVILY_API_KEY"] = "<Your Tavily API Key>"
os.environ["FIRECRAWL_API_KEY"] = "<Your FireCrawl API Key>"
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
)

server_address = "<The address you want to bind the server to. The server should be running the `HTTP_server.py` script."
port = 8080
