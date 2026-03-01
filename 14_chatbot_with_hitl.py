from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated,List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests


load_dotenv()

model  = ChatOpenAI()

class Chat(TypedDict):

    messages : Annotated[List[BaseMessage],add_messages]


@tool
def stock_price(symbol:str)->dict:

    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"

    response = requests.get(url)
    return response.json

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    NOTE: This is a mock implementation:
    - No real brokerage API is called.
    - It simply returns a confirmation payload.
    """

    decision = interrupt({
        'type': 'approval',
        'reason': 'User wants to purchase the share',
        'question': '',
        'instruction': 'Approved, yes/no'
    })

    if decision['approval'] == 'yes':
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }

tool_kit = [stock_price,purchase_stock]

model_with_tool = model.bind_tools(tool_kit)

def chat_node(state: Chat):

    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = model_with_tool.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tool_kit)

graph = StateGraph(Chat)

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

config = {'configurable':{'thread_id':'thread-chat'}}

if __name__  == '__main__':

    while True :

        user_input = input('You :')
        if user_input.lower() in ['bye','exit','quit']:
            break
        response = chatbot.invoke({'messages':[HumanMessage(content = user_input)]},config=config)

        interupt = response.get("__interrupt__", [])

        if interupt:

            ans = input('System Message : Kindly grant approval (yes/no)')

            response = chatbot.invoke(Command(resume={'approval':ans}),config=config)

        print('AI Message :',response['messages'][-1].content)