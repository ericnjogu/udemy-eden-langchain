from dotenv import load_dotenv

load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool

MAX_ITERATIONS = 10;
MODEL = "gemma3:270m"

@tool
def get_product_prices(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f" executing get_product_prices for {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0);

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """apply a discount tier to a price and return the final price
    available tiers are silver, gold, platinum
    """
    discount_percentages = {"silver": 5, "gold": 10, "platinum": 15}
    return price * (1 - discount_percentages.get(discount_tier, 0) / 100)

def run_agent(question: str):
    pass

if __name__ == "__main__":
    print("running agent...")
    result = run_agent("What is the price of a monitor after applying a gold discount?")