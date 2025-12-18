# %%
from typing import Annotated, Sequence
from langchain.schema import BaseMessage
from langgraph.graph import add_messages
from typing import TypedDict

# %%
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# %%
from langchain_community.tools import DuckDuckGoSearchRun

# %%
search = DuckDuckGoSearchRun()

# %%
from langchain_community.tools import Tool

# %%
search_tool = Tool(
    name = "duck duck go search",
    description = "use to search government sites only. Government sites have .gov in the url. Gather only latest information.",
    func = search.run
)

# %%
from langchain_ollama.llms import OllamaLLM
llm = OllamaLLM(model = "llama3.1")

# %%
from langchain_core.output_parsers import StrOutputParser

# %%
from langchain import hub

# %%
def generate(state:AgentState):
    print("GENERATE")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = prompt | llm | StrOutputParser()
    
    response = rag_chain.invoke({"context" : docs, "question": question})
    return {"messages": messages + [response]}

# %%
from langgraph.prebuilt import ToolNode

# %%
from langgraph.graph import StateGraph

# %%
workflow = StateGraph(AgentState)
gov_search = ToolNode([search_tool])
workflow.add_node("gov_search", gov_search)
workflow.add_node("generator", generate)


# %%
from langgraph.graph import START, END

# %%
workflow.add_edge(START, "gov_search")
workflow.add_edge("gov_search", "generator")
workflow.add_edge("generator", END)

# %%
app = workflow.compile()

# %%
from IPython.display import Image, display

try:
    display(Image(app.get_graph(xray = True).draw_mermaid_png()))
except Exception:
    pass

# %%
from langchain_core.messages import HumanMessage, AIMessage

# %%
app.invoke({
    "messages": [
        AIMessage(content="You are a visa assistant."),
        HumanMessage(content="What is F1 visa fees USA")
    ]
})


