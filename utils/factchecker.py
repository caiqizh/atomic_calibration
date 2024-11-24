import numpy as np
import json
from wiki_retrieval import DocDB, Retrieval
from wild_retrieval import WildRetrieval
from openai import OpenAI
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import os
from transformers import GPT2Tokenizer

INSTUCT_FACTCHECK = """Your task is to fact-check the following statements starting with ###. These statements are extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

For each statement, assign a veracity label: 'S' means the statement is supported by your knowledge and is factually correct; 'NS' means the statement is not supported or you are unsure based on your knowledge. Pay close attention to numbers, dates, and other details.

To assess a statement, first analyze it and then add the veracity label enclosed in dollar signs ($). The format is as follows:

### Lebron James is a basketball player. Analysis: Lebron James is an American basketball player, so this is correct. $S$
### Obama was the 46th president of the United States. Analysis: Obama was the 44th president of the United States, so this is incorrect. $NS$
### Jackie Chan was born on April 7, 1955. Analysis: Jackie Chan was born on April 7, 1954, so this is incorrect. $NS$
### ECC9876 is a great place to visit. Analysis: To the best of my knowledge, there is no evidence saying ECC9876 is a great place to visit. $NS$

The statements to evaluate are as follows:
{atomic_facts_string}
"""

INSTUCT_FACTCHECK_RAG_ALL = """Your task is to fact-check the following statements starting with ###. These statements are extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

For each statement, assign a veracity label: 'S' means the statement is factually correct; 'NS' means the statement is factually wrong. Pay close attention to numbers, dates, and other details. You can refer to the provided knowledge source to verify the statements.

To assess a statement, first analyze it and then add the veracity label enclosed in dollar signs ($). The format is as follows:

### Lebron James is a basketball player. Analysis: Lebron James is an American basketball player, so this is correct. $S$
### Obama was the 46th president of the United States. Analysis: Obama was the 44th president of the United States, so this is incorrect. $NS$
### Jackie Chan was born on April 7, 1955. Analysis: Jackie Chan was born on April 7, 1954, so this is incorrect. $NS$

The statements to evaluate are as follows:
{atomic_facts_string}

The knowledge source for this fact-check is as follows:
{retrieved_evidence}
"""

INSTUCT_FACTCHECK_RAG_TOP = """Your task is to fact-check the following statements starting with ###. These statements are extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

For each statement, assign a veracity label: 'S' means the statement is factually correct; 'NS' means the statement is factually wrong. Pay close attention to numbers, dates, and other details. 

For each statement, you are provided with some evidence to help you verify the statement. It is not guraranteed that the evidence is relevant to the statement. If not, you can ignore the evidence and make your own judgement based on your knowledge.

To assess a statement, add the veracity label enclosed in dollar signs ($). The format is as follows:

Input: 
Statement: Lebron James is a basketball player.

Evidence: Lebron James is an American professional basketball player for the Los Angeles Lakers of the National Basketball Association (NBA). He is widely considered to be one of the greatest basketball players in NBA history.

Statement: Obama was the 46th president of the United States.

Evidence: Barack Hussein Obama II (born August 4, 1961) is an American politician who served as the 44th president of the United States from 2009 to 2017. As a member of the Democratic Party, he was the first African-American president in U.S. history.

Statement: Jackie Chan was born on April 7, 1955. 

Evidence: He then made his big screen debut in the role of a Jehovah's Witness who befriends an unemployed professor in the independent film Host and Guest, which traveled the international film festival circuit.

Output:
### Lebron James is a basketball player. $S$
### Obama was the 46th president of the United States. $NS$
### Jackie Chan was born on April 7, 1955. $NS$

Your output should strictly follow the format above. Do not include any additional information or text. The statements and evidences to evaluate are as follows:
{atomic_fact_with_evidence}
"""


class FactChecker(object):
    '''
    
    '''
    def __init__(self, max_evidence_length):
        super().__init__()
        self.llm_token_count = 0
        self.openai_cost = 0
        self.db = {}
        self.retrieval = {}
        self.max_evidence_length = max_evidence_length

    def truncate_text(self, text):
        """Truncate the text to a maximum length."""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
        if len(tokens) > self.max_evidence_length:
            tokens = tokens[:self.max_evidence_length]
            text = tokenizer.decode(tokens)
        return text

    def build_enwiki_evidence(self, knowledge_source="enwiki",
                              db_path='../factcheck_cache/enwiki-20230401.db',
                              data_path='', cache_path='../factcheck_cache/retrieval-enwiki-20230401.json',
                              embed_cache_path='../factcheck_cache/retrieval-enwiki-20230401.pkl',
                              batch_size=256):
        """Build evidence from English Wikipedia."""
        self.db[knowledge_source] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[knowledge_source] = Retrieval(self.db[knowledge_source], cache_path, embed_cache_path, batch_size=batch_size)

    def build_google_search(self, knowledge_source="google_search", db_path="../factcheck_cache/wildhallu.db",
                            data_path=""):
        """Build evidence from Google Search."""
        self.retrieval[knowledge_source] = WildRetrieval(db_path=db_path)
        print("Finished building Google Search evidence.")

    @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(20))
    def get_completion(self, user_prompt: str=""):
        """Get completion from the GPT model."""
        client = OpenAI(base_url="https://gptproxy.llmpaas.woa.com/v1", api_key=os.getenv("OPENAI_API_KEY"))
        for i in range(5):
            try:
                response = client.chat.completions.create(model="gpt-4o", messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ])
                return response.choices[0].message.content
            except Exception as e:
                print(f'ERROR: {str(e)}')
                print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
                time.sleep(2 ** (i + 1))
        return None

    def get_prompt_zero_all(self, topic, fact_list, dataset, evidence_type):
        atomic_facts_string = "### " + "\n### ".join(fact_list) + "\n\n"
        instruct = ""
        retrieved_evidence_string = ""
        if evidence_type == "zero":
            instruct = INSTUCT_FACTCHECK.format(atomic_facts_string=atomic_facts_string)
        elif evidence_type == "all":
            if dataset == "longfact":
                raise ValueError("LongFact does not support evidence retrieval")
            elif dataset == "bios":
                passages = self.retrieval['enwiki'].get_all_passages(topic)
                self.retrieval['enwiki'].save_cache()
                retrieved_evidence_string = "".join([i["text"] for i in passages])
            else:  # dataset == "wildhallu"
                passages = self.retrieval['google_search'].get_all_passages(topic)
                retrieved_evidence_string = "".join([i for i in passages])
            instruct = INSTUCT_FACTCHECK_RAG_ALL.format(atomic_facts_string=atomic_facts_string, retrieved_evidence=retrieved_evidence_string)
        return self.truncate_text(instruct)
    
    def get_prompt_topk(self, topic, fact_list, dataset, topk=1):
        atomic_fact_with_evidence = ""
        if dataset == "longfact":
            # For long-fact, we provide the entity linking results in a separate file.
            # The entity linking results are used to retrieve evidence from the Wikipedia.
            raise ValueError("LongFact does not support evidence retrieval at the moment.")
        if dataset in ["bios", "wildhallu"]:
            for fact in fact_list:
                if dataset == "bios":
                    passages = self.retrieval['enwiki'].get_passages(topic, fact, topk)
                else:
                    try:
                        passages = self.retrieval['google_search'].get_gtr_passages(topic, fact, topk)
                    except Exception as e:
                        if "CUDA out of memory" in str(e):
                            print(f"Error in retrieving evidence for fact: {fact} using GTR retrieval. Error: {e}")
                            print(f"Trying BM25 retrieval for fact: {fact}.")
                            passages = self.retrieval['google_search'].get_bm25_passages(topic, fact, topk)

                evidence = "".join([i["text"] if dataset == "bios" else i for i in passages])
                atomic_fact_with_evidence += f"Statement: {fact}\n\nEvidence: '''{evidence}'''\n\n"
        instruct = INSTUCT_FACTCHECK_RAG_TOP.format(atomic_fact_with_evidence=atomic_fact_with_evidence)
        print(instruct)
        return self.truncate_text(instruct)
    
    def get_prompt(self, topic, fact_list, dataset, evidence_type):
        if evidence_type in ["zero", "all"]:
            # print(fact_list)
            return self.get_prompt_zero_all(topic, fact_list, dataset, evidence_type)
        elif evidence_type == "topk":
            return self.get_prompt_topk(topic, fact_list, dataset)
        else:
            raise ValueError("Invalid evidence type")   
        
    def get_veracity_labels(self, topic="", atomic_facts=[], knowledge_source="", evidence_type="zero"):
        """
        Get veracity labels for the given atomic facts.
        We need to try multiple times because the GPT model may fail to respond.
        """

        for attempt in range(3):
            try:
                user_prompt = self.get_prompt(topic, atomic_facts, knowledge_source, evidence_type=evidence_type)
                completion = self.get_completion(user_prompt)
                atomic_responses = completion.split("\n### ")
                atomic_responses = [x for x in atomic_responses if x]
                gpt_labels = [response.split("$")[1] for response in atomic_responses]
                assert len(atomic_facts) == len(gpt_labels)
                return gpt_labels, completion
            except Exception as e:
                print(e)
                print(f'Retrying ({attempt + 1}/3), wait for {2 ** (attempt + 1)} sec...')
                time.sleep(2 ** (attempt + 1))
        return [], ""

        
if __name__ == "__main__":
    
    os.environ["OPENAI_API_KEY"] = "zT3gHymzPd4rLtwURgpkUp1sLHpR7lyt"

    # topic = "University of Cambridge"
    # fact_list = ["Cambridge University is a public research university in Cambridge, England.",
    #              "Cambridge University was founded in 1209.",
    #              "Cambridge University has 29 colleges.",
    #              "Cambridge University was founed in 1309.",
    #              "Cambridge University has 31 colleges."
    #              ]
    
    # topic = "Kang Ji-hwan"
    # fact_list = ["Kang Ji-hwan is a South Korean actor.", "Kang Ji-hwan is a South Korean singer.", "Kang Ji-hwan was born on August 6, 1977.", "Kang Ji-hwan was born in Seoul, South Korea.", "Kang Ji-hwan began his career as a member of the boy band \"Sechs Kies\".", "Kang Ji-hwan began his career in the late 1990s.", "The boy band \"Sechs Kies\" disbanded in 2000.", "Kang Ji-hwan transitioned to acting after his music career.", "Kang Ji-hwan made his acting debut in the 2001 film \"Break Out\".", "Kang Ji-hwan gained recognition for his role in \"Lovers in Paris\" in 2004.", "Kang Ji-hwan gained recognition for his role in \"Be Strong Geum-soon\" in 2005.", "Kang Ji-hwan has appeared in \"The World That They Live In\" in 2008.", "Kang Ji-hwan has appeared in \"The Fugitive Plan B\" in 2010.", "Kang Ji-hwan has appeared in \"Myung-wol the Spy\" in 2013.", "Kang Ji-hwan has appeared in \"My ID is Gangnam Beauty\" in 2018.", "Kang Ji-hwan received the Best Actor award at the 2005 SBS Drama Awards.", "Kang Ji-hwan received the award for his role in \"Be Strong Geum-soon\".", "Kang Ji-hwan is known for his versatility as an actor.", "Kang Ji-hwan has played a wide range of characters.", "Kang Ji-hwan's characters span various genres.", "Kang Ji-hwan is also a talented singer.", "Kang Ji-hwan has released several solo albums.", "Kang Ji-hwan is married to actress Park Ha-sun.", "Kang Ji-hwan has two children with Park Ha-sun."]
    
    fc = FactChecker(max_evidence_length=100000)
    fc.build_google_search()

    # print("Checking bios without evidence")
    # veracity_labels = fc.get_veracity_labels(topic, fact_list, "bios", evidence_type="zero")
    # for fact, label in zip(fact_list, veracity_labels):
    #     print(fact, label)
    # print("*"*50)

    # print("Checking bios with evidence")
    # fc.build_enwiki_evidence()
    # veracity_labels = fc.get_veracity_labels(topic, fact_list, "bios", evidence_type="topk")
    # for fact, label in zip(fact_list, veracity_labels):
    #     print(fact, label)
    # print("*"*50)

    # print("Checking wildhallu without evidence")
    # veracity_labels = fc.get_veracity_labels(topic, fact_list, "wildhallu", evidence_type="zero")
    # for fact, label in zip(fact_list, veracity_labels):
    #     print(fact, label)
    # print("*"*50)

    # print("Checking longfact without evidence")
    # veracity_labels = fc.get_veracity_labels(topic, fact_list, "longfact", evidence_type="zero")
    # for fact, label in zip(fact_list, veracity_labels):
    #     print(fact, label)
    # print("*"*50)

    # print("Checking wildhallu with evidence")
    # fc.build_google_search(db_path = "../factcheck_cache/wildhallu.db")
    # veracity_labels = fc.get_veracity_labels(topic, fact_list, "wildhallu", evidence_type="topk")
    # for fact, label in zip(fact_list, veracity_labels):
    #     print(fact, label)

    topic = "ESP8266"
    fact_list = ['The ESP8266 is a series of low-cost Wi-Fi microchips.', 'The ESP8266 has a full TCP/IP stack.', 'The ESP8266 has microcontroller capability.', 'Espressif Systems produces the ESP8266.', 'Espressif Systems is in China.', 'The ESP8266 chips are popular among hobbyists.', 'The ESP8266 chips are popular among DIY enthusiasts.', 'The ESP8266 chips are used for building various Internet of Things (IoT) projects.', 'The ESP8266 chips can be programmed using the Arduino IDE.', 'The ESP8266 chips can be programmed using the Micropython programming language.', 'The ESP8266 chips allow users to create Wi-Fi-enabled projects.', 'The Wi-Fi-enabled projects can send data.', 'The Wi-Fi-enabled projects can receive data.', 'The Wi-Fi-enabled projects can control other devices.', 'The Wi-Fi-enabled projects can interface with cloud services.', 'The ESP8266 comes in different versions.', 'The ESP8266 versions have varying specifications.', 'The ESP-01 is a version of the ESP8266.', 'The ESP-07 is a version of the ESP8266.', 'The ESP-12 is a version of the ESP8266.', 'The ESP32 is a version of the ESP8266.', 'The ESP8266 has a small size.', 'The ESP8266 has low power consumption.', 'The ESP8266 has an affordable price.', 'The ESP8266 is ideal for building simple automation projects.', 'The ESP8266 is ideal for building home security systems.', 'The ESP8266 is ideal for building weather stations.', 'The ESP8266 is used for many other IoT applications.']
    
    veracity_labels = fc.get_veracity_labels(topic=topic, 
                                                 atomic_facts=fact_list, 
                                                 knowledge_source="wildhallu", 
                                                 evidence_type="zero")