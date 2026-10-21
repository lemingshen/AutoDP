import prompt, tool_list

from langgraph.graph import END
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage

model_information_database = Chroma(
    collection_name="user_interface_model_information",
    embedding_function=tool_list.embeddings,
    persist_directory="./chroma_langchain_db",
)
model_information_graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = model_information_database.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# step 1: generate an AIMessage that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = tool_list.model.bind_tools([retrieve])

    first_retrieve = True
    for message in reversed(state["messages"]):
        if message.type == "tool":
            first_retrieve = False
            break

    if not first_retrieve:
        state["messages"].append(HumanMessage(content=prompt.reinvoke_retrieval_prompt))

    response = llm_with_tools.invoke(state["messages"])

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
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
        SystemMessage(prompt.model_information_system_prompt.format(retrieved_context))
    ] + conversation_messages

    # Run
    response = tool_list.model.invoke(response_generation_prompt)
    return {"messages": [response]}


model_information_graph_builder.add_node(query_or_respond)
model_information_graph_builder.add_node(tools)
model_information_graph_builder.add_node(generate)

model_information_graph_builder.set_entry_point("query_or_respond")
model_information_graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: "generate", "tools": "tools"},
)
model_information_graph_builder.add_edge("tools", "query_or_respond")
model_information_graph_builder.add_edge("generate", END)
model_information_graph = model_information_graph_builder.compile()
