from promptflow import tool
from chat_with_pdf.main import download

@tool
def download_tool(url: str, env_ready_signal: str) -> str:
    return download(url)