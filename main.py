from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langsmith import traceable

MAX_ITERATIONS = 10;
MODEL = "ollama:gemma4:12b"

@tool
def get_product_prices(product: str) -> float:
    """Look up the price of a product in the catalog
    returns -1 if the product is not present
    """
    print(f" executing get_product_prices for {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    if product not in prices:
        print(f"product '{product}' not present in catalog")
        return -1;
    return prices.get(product, 0);

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """apply a discount tier to a price and return the final price
    available tiers are silver, gold, platinum
    """
    discount_percentages = {"silver": 5, "gold": 10, "platinum": 15}
    return price * (1 - discount_percentages.get(discount_tier, 0) / 100)

@traceable#(name="react agent loop")
def run_agent(question: str):
    tools = [get_product_prices, apply_discount]
    tools_dict = {t.name: t for t in tools}
    # alternative model - openai:gpt-5
    llm = init_chat_model(f"{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content="you are a helpful shopping assistant"
            "you have access to a product catalog tool and a discount tool"
        ),
        HumanMessage(content=question)
    ]
    
    for iteration in range(0, MAX_ITERATIONS):
        print(f"iteration {iteration}")
        ai_message = llm_with_tools.invoke(messages)
        # print(f"ai_message: {ai_message}")
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"final answer: {ai_message.content}")
            return ai_message.content
        # process only the first tool call\
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")
        print(f"Tool selected: {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"tool '{tool_name}' not found")
        observation = tool_to_use.invoke(tool_args)
        print(f"observation: {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_id))
    
    print("MAX iterations reached")
    return None

if __name__ == "__main__":
    print("running agent...")
    result = run_agent("What is the price of a keyboard after applying a gold discount?")