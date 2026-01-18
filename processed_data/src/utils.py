import json
import time
from openai import OpenAI
import openai
import random
# from config import OPENAI_API_KEY, QWEN_API_KEYS,QWEN_BASE_URL,OPENAI_BASE_URL



# Example configuration file for Faico
# Copy this file to config.py and fill in your actual values

# OpenAI API Configuration
OPENAI_API_KEY = [
            "your_openai_api_key_1",
            "your_openai_api_key_2",
            "your_openai_api_key_3"
        ]
OPENAI_BASE_URL = "https://api.openai.com/v1"  # Or your proxy URL

# Qwen API Configuration
QWEN_API_KEYS = [
            "sk-9fea050c0f9f4bdbaa3e8678b9f50283",
            "sk-c38315c526a943a898836b5572a039e0",
            "sk-d1b47c76e3c24f8fb9568361af2021bd",
            "sk-22482e780f4b43b0aa64296f2b518c93"
        ]
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # Official Qwen API URL

def run_llm_inference(prompt, temperature, max_tokens, engine="qwen-plus"):
    print("LLM engine:",engine)
    if "qwen" in engine or "deepseek" in engine:
        apikey = random.choice(QWEN_API_KEYS)
        client = OpenAI(
            api_key=apikey,
            base_url=QWEN_BASE_URL
        )
    elif "gpt" in engine :
        key = random.choice(OPENAI_API_KEY)
        client = OpenAI(api_key=key,base_url=OPENAI_BASE_URL)


    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    
    max_retries = 1
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if engine == "qwen-plus":
                response = client.chat.completions.create(
                        model=engine,
                        messages = messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed = 0,
                        extra_body={"enable_thinking": False}
                        )
            else:
                response = client.chat.completions.create(
                            model=engine,
                            messages = messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
            result = response.choices[0].message.content

            return result
            
        except openai.RateLimitError:
                print(f"Rate limit exceeded, retry {retry_count+1}/{max_retries}")
        except Exception as e:
            print(f"Unexpected error: {e}, retry {retry_count+1}/{max_retries}")
        
        retry_count += 1
        time.sleep(2)
        
    print(f"Failed after {max_retries} retries")    
    return ""
