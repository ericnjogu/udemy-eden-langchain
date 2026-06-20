from dotenv import load_dotenv
import ollama

load_dotenv()

from langchain.chat_models import init_chat_model
from langsmith import traceable

MAX_ITERATIONS = 10;
MODEL = "gemma4:12b"
# MODEL = "qwen3.5:2b"

@traceable
def get_product_prices(product: str) -> float:
    """Look up the price of a product in the catalog
    Args:
        product: product name
    Returns:
        price, else -1 if the product is not present
    """
    print(f" executing get_product_prices for {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    if product not in prices:
        print(f"product '{product}' not present in catalog")
        return -1;
    return prices.get(product, 0);

@traceable
def apply_discount(price: float, discount_tier: str) -> float:
    """apply a discount tier to a price
    Args:
        discount_tier: available tiers are silver, gold, platinum
        price: product price
    Returns:
        the final price
    """
    discount_percentages = {"silver": 5, "gold": 10, "platinum": 15}
    return price * (1 - discount_percentages.get(discount_tier, 0) / 100)

@traceable(name="ollama chat", type="llm")
def ollama_chat_traced(messages, tools):
    return ollama.chat(model=MODEL, tools=tools, messages=messages)

@traceable#(name="react agent loop")
def run_agent(question: str):
    tools = [get_product_prices, apply_discount]
    tools_dict = {t.__name__: t for t in tools}
    #return None
    # alternative model - openai:gpt-5

    messages = [
        {
        "role": "system",
        "content": "you are a helpful shopping assistant"
            "you have access to a product catalog tool and a discount tool"
        ,},
        {"role": "user", "content": question}
    ]
    
    for iteration in range(0, MAX_ITERATIONS):
        print(f"iteration {iteration}")
        ai_message = ollama_chat_traced(messages, tools).message
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"final answer: {ai_message.content}")
            return ai_message.content
        print(f"tool calls {tool_calls}")
        # process only the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        print(f"Tool selected: {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"tool '{tool_name}' not found")
        observation = tool_to_use(**tool_args)
        print(f"observation: {observation}")
        messages.append(ai_message)
        messages.append({
            "role":"tool",
            "content": str(observation)
        })
    
    print("MAX iterations reached")
    return None

if __name__ == "__main__":
    print("running agent...")
    result = run_agent("What is the price of a laptop after applying a silver discount?")