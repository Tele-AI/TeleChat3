from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="{{YOUR_MODEL_PATH}}",
    messages=[
        {"role": "user", "content": "hi"},
    ],
    temperature=0.6, #recommend
    max_tokens=16384, #recommend for general chat, we recommend 30000 for math, logic and code inference
    extra_body={
        "repetition_penalty": 1.05, #recommend
        "skip_special_tokens": False, #please do not modify!
        "spaces_between_special_tokens": False, #please do not modify!
    },
)

print("Chat response:", chat_response.choices[0].message)
