import openai
import re
import backoff

api_key = 'your own key'
openai.api_key = api_key


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout, openai.error.ServiceUnavailableError))
def askChatGPT(prompt, model_name: str = "gpt-3.5-turbo-0125"):
    messages = [{"role": "user", "content": prompt}]
    # print(messages[0]['content'])
    response = openai.ChatCompletion.create(model=model_name, messages=messages)
    # response = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=0)
    return response['choices'][0]['message']['content']
