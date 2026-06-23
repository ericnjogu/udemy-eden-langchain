from dotenv import load_dotenv
import ollama
import inspect
import re

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
    return float(price) * (1 - discount_percentages.get(discount_tier, 0) / 100)

def get_tool_descriptions():
    descriptions = []
    for t in [get_product_prices, apply_discount]:
        original_function = getattr(t, "__wrapped__", t)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(t) or ""
        descriptions.append(f"{t.__name__}{signature}-{docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions()
tool_funcs = [apply_discount, get_product_prices]
tools = {t.__name__: t for t in tool_funcs}
tool_names = ", ".join(tools.keys())

# https://smith.langchain.com/hub/hwchase17/react
react_prompt = f"""
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""

@traceable(name="ollama chat", type="llm")
def ollama_chat_traced(messages, options):
    return ollama.chat(model=MODEL, messages=messages, options=options)

@traceable#(name="react agent loop")
def run_agent(question: str):
    print(f"Question: {question}")
    prompt = react_prompt.format(
        question=question,
        tool_descriptions=tool_descriptions,
        tool_names=tool_names,
    )
    scratchpad = ""
    for iteration in range(0, MAX_ITERATIONS):
        print(f"---- iteration {iteration}")
        full_prompt = prompt + scratchpad
        response = ollama_chat_traced(
            messages=[{"role":"user", "content": full_prompt}],
            options = {"stop": ["\nObservation"], "temperature": 0}
        )
        output = response.message.content
        print(f"LLM output:\n{output}")
        print(f"[Parsing] Looking for final answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"[Parsed] Final answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"{final_answer}")
            return final_answer
        print(f" [Parsing] looking for action and action iput in LLM output...")
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        if not action_match or not action_input_match:
            print("[parsing] Error: could not parse action/action-input from LLM output")
            break
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"[Tool selected] {tool_name} with args: {tool_input_raw}")
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        print(f"raw_args: {raw_args}")
        # gemma produces action input as JSON - gemma4:12b
        # could have imported json and parsed this
        args = [x.split(":", 1)[-1].strip().strip("'\"}") for x in raw_args]
        # possibly works for qwen
        # args = [x.split(":", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"[Tool executing] with args {args}")
        if tool_name not in tools:
            observation = f"Error: Tool {tool_name} not found. Available tools {tools.keys()}"
        else:
            observation = str(tools[tool_name](*args))
            # if isinstance (args[0], dict):
            #     print(f"using values to call dict args {args[0].values()}")
            #     observation = str(tools[tool_name](*args[0].values()))
            # else:
            #     print(f"using args to invoke tool {args}")
            #     observation = str(tools[tool_name](*args))  
        print(f"[Tool result]: {observation}")
        scratchpad += f"{output}\nObservation: {observation}\nThought:"
    print("ERROR: max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("running agent...")
    result = run_agent("What is the price of a laptop after applying a silver discount?")