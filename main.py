import time

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    # print(os.getenv("OPEN_API_KEY"))
    info: str = """
    Jet Li Lianjie (courtesy name Yangzhong; born 26 April 1963)[2] is a martial artist, actor, and philanthropist.
     With a career spanning more than forty years, he is regarded as one of the greatest 
     Chinese film stars and one of the greatest martial artists in the history of cinema.[
     3][4][5] His film career in Asia is credited with reviving Hong Kong kungfu films[6][7] 
     as well as Shaolin Temple.[8] Li was trained as a wushu athlete at the Beijing Shichahai Sports 
     School and went on to win multiple national championships with the Beijing Wushu 
     Team between 1974 and 1979.[6] After his retirement from the sport in 1979, 
     he made his acting debut with the Hong Kong film Shaolin Temple (1982),[7] 
     a runaway success followed by two sequels in 1984 and 1986. Li established himself as a 
     leading action star with the Once Upon a Time in China series (1991–1993), in which he 
     portrayed Chinese folk hero Wong Fei-hung, followed by Born to Defence (1988), 
     which is his directorial debut, Swordsman II (1992), Fong Sai-yuk (1993), 
     Fist of Legend (1994), High Risk (1995), Black Mask (1996), and Hitman (1998).[9] 
    """
    summary_template: str = """
    Given the following information {info} about a person from a biography, please list the following attributes:
    - short summary
    - two interesting facts about the person
    """
    prompt: PromptTemplate = PromptTemplate(
        template=summary_template,
        input_variables=["info"]
    )
    #model_name = "gpt-5.4-mini"
    #model: ChatOpenAI = ChatOpenAI(model=model_name, temperature=0)
    model_name = "gemma3:270m"
    model: ChatOllama = ChatOllama(model=model_name, temperature=0)
    chain: RunnableSequence = prompt | model
    start = time.perf_counter()
    response: str = chain.invoke({"info": info})
    elapsed = time.perf_counter() - start
    print(f"chain.invoke took {elapsed:.2f}s for {model_name}")
    print(response.content)


if __name__ == "__main__":
    main()
