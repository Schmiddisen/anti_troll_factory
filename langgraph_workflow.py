from langgraph.graph import StateGraph, END
from typing import Optional, TypedDict, Dict, Any
from langchain.tools import tool
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Define the state type
class GraphState(TypedDict):
    humanized_response: Optional[str] = None
    reponse_writer_answer: Optional[str] = None

# Define the tools
@tool
def check_climate_info(query: str) -> str:
    """
    This tool can be used to check climate information from peer-reviewed papers. It is a similarity search RAG. Use this tool multiple times if needed.
    """
    pass

# Define the nodes
def response_writer(state: GraphState):
    """
    Node for writing a factual response based on the conversation thread.
    """
    
    conversation_history = state["conversation_history"]
    prompt_template = """You are an expert debater. Based on this conversation thread: {conversation_history}, write a factual response that proves them wrong.
    The response should be concise and to the point. Do not include any personal opinions or feelings. Only use facts.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    response_writer = create_react_agent(
        model=ChatGoogleGenerativeAI(model_name=""),
        tools=[check_climate_info],
        prompt=prompt
    )
    
    response = response_writer.invoke()
    
    return {"reponse_writer_answer":response["messages"][-1]["content"]}

def humanizer(state: GraphState):
    """
    Node for humanizing the tweet reply.
    """
    
    # TODO use prompttemplate correctly
    factual_reply = state["reponse_writer_answer"]
    conversation_history = state["conversation_history"]
    
    prompt_template = f"""You are a social media expert. You are given a tweet reply: {tweet_reply} and its context: {conversation_history}. 
    Change the style in a human way that matches the tweeting style of the previous conversation, and how people generally answer in tweets.
    The response should be engaging and relatable. Use emojis and slang where appropriate. Match the language of the conversation.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    response_writer = create_react_agent(
        model=ChatGoogleGenerativeAI(model_name=""),
        tools=[],
        prompt=prompt
    )
    
    response = response_writer.invoke()

    return {"humanized_response": response["messages"][-1]["content"]}

def verifier(state: GraphState):
    """
    Node for verifying the tweet reply.
    """
    tweet_reply = state["keys"].get("humanized_response", "")
    
    humanized_tweet = state["tweet_reply"]
    conversation_history = state["conversation_history"]
    
    prompt_template = f"""You are a fact checker. You are given a tweet reply: {tweet_reply} and the context surrounding it: {conversation_history}. 
    Check if the given reply is factual and gets its point across. You are given the check_climate_info tool to receive peer reviewed paper info.
    If the reply is not factual, rewrite it to be factual.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    response_writer = create_react_agent(
        model=ChatGoogleGenerativeAI(model_name=""),
        tools=[],
        prompt=prompt
    )
    
    response = response_writer.invoke()
    
    return {"verified_response": response["messages"][-1]["content"]}

def tweet_node(state: GraphState):
    """
    Node for tweeting the reply.
    """
    verified_response = state["keys"].get("verified_response", "")
    # Logic to tweet the reply
    # This is a placeholder, replace with actual implementation
    print(f"Tweeting: {verified_response}")
    return {}

# Define the graph
graph = StateGraph(GraphState)

# Add the nodes
graph.add_node("response_writer", response_writer)
graph.add_node("humanizer", humanizer)
graph.add_node("verifier", verifier)
graph.add_node("tweet_node", tweet_node)

# Add the edges
graph.add_edge("response_writer", "humanizer")
graph.add_edge("humanizer", "verifier")
graph.add_edge("verifier", "tweet_node") # TODO change this to a conditional edge should the verifier find that it is wrong.
graph.add_edge("tweet_node", END)

# Set the entry point
graph.set_entry_point("response_writer")

# Compile the graph
chain = graph.compile()

# Example usage
if __name__ == "__main__":
    state = {"keys": {"conversation_history": "This is a dummy conversation history."}}
    result = chain.invoke(state)
    print(result)
