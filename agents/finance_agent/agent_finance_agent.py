# agents/finance/agent.py
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from langchain.agents import create_react_agent
from agents_llm_setup import llm
from agents.finance_agent.agent_finance_tool import finance_tools
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate

# --- Prompt for React Agent ---
finance_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are a Finance Agent. Use the provided tools to process invoices, update ledgers, detect anomalies, and enforce policies.

Tools:
{tools}

Available tool names: {tool_names}

Use the following format:
User Query: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

User Query: {input}
{agent_scratchpad}
"""
)
# --- Create Finance Agent ---
finance_agent_instance = create_react_agent(
    llm=llm,
    tools=finance_tools,
    prompt=finance_prompt
)

finance_agent = AgentExecutor.from_agent_and_tools(
    finance_agent_instance,
    tools=finance_tools,
    handle_parsing_errors=True,
    verbose=True  
)

# --- Test ---
if __name__ == "__main__":
    res = finance_agent.invoke({"input": "What is the refund policy ? "})
    print(res)
    res = finance_agent.invoke({"input": "How many invoices do we have this month?"})
    print(res)
    res=finance_agent.invoke({"input": "List suspicious payments over $5000."})

    print(res)
