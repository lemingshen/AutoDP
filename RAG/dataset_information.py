import prompt, tool_list

from langgraph.graph import END
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

dataset_information_database = Chroma(
    collection_name="dataset_information",
    embedding_function=tool_list.embeddings,
    persist_directory="./chrome_langchain_db",
)
dataset_information_graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = dataset_information_database.similarity_search(query, k=10)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# step 1: generate an AIMessage that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = tool_list.model.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve, tool_list.web_search_tool])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    retrieved_context = "\n\n".join(document.content for document in tool_messages)
    print("Retrieved the following information from the database:")
    print("\033[33m{}\033[0m".format(retrieved_context))

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    response_generation_prompt = [
        SystemMessage(
            prompt.dataset_information_system_prompt.format(retrieved_context)
        )
    ] + conversation_messages

    # Run
    response = tool_list.model.invoke(response_generation_prompt)
    return {"messages": [response]}


dataset_information_graph_builder.add_node(query_or_respond)
dataset_information_graph_builder.add_node(tools)
dataset_information_graph_builder.add_node(generate)

dataset_information_graph_builder.set_entry_point("query_or_respond")
dataset_information_graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
dataset_information_graph_builder.add_edge("tools", "generate")
dataset_information_graph_builder.add_edge("generate", END)
dataset_information_graph = dataset_information_graph_builder.compile()
