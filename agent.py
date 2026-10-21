import tool_list, prompt
import RAG.model_optimization
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig, chain


common_search_agent = tool_list.model.bind_tools([tool_list.web_search_tool])

# %% device information search agent
device_information_search_agent = prompt.device_spec_final_prompt | common_search_agent


@chain
def device_information_search_chain(user_input: str, config: RunnableConfig):
    input_ = {"input": user_input}
    ai_msg = device_information_search_agent.invoke(input_, config=config)
    tool_msgs = tool_list.web_search_tool.batch(ai_msg.tool_calls, config=config)

    return device_information_search_agent.invoke(
        {**input_, "searched_results": [ai_msg, *tool_msgs]}, config=config
    )


# %% General one-time search agent without chat history
general_search_agent = create_react_agent(tool_list.model, [tool_list.web_search_tool])


# %% local deployment for memory optimization agent
local_deployment_agent = create_react_agent(
    tool_list.model,
    [tool_list.web_search_tool],
    checkpointer=tool_list.agent_memory,
)
local_deployment_agent_configuration = {
    "configurable": {"thread_id": "local_deployment"}
}

# %% agent for optimization method generation
optimization_method_generation_agent = create_react_agent(
    tool_list.model,
    [tool_list.web_search_tool],
    checkpointer=tool_list.agent_memory,
)
optimization_method_generation_agent_configuration = {
    "configurable": {"thread_id": "optimization_method_generation"}
}


# %% search space determination and hyperparameter configurations
local_optimization_agent = create_react_agent(
    tool_list.model,
    [tool_list.web_search_tool, RAG.model_optimization.retrieve],
    checkpointer=tool_list.agent_memory,
)
local_optimization_agent_configuration = {
    "configurable": {"thread_id": "local_optimization"}
}

# %% agent for optimized code local deployment
edge_model_deployment_agent = create_react_agent(
    tool_list.model,
    [tool_list.web_search_tool],
    checkpointer=tool_list.agent_memory,
)
edge_model_deployment_agent_configuration = {
    "configurable": {"thread_id": "edge_optimization"}
}
