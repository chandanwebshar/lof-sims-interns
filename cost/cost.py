import time
import requests
import json
import PyPDF2

OPENROUTER_API_KEY = "xxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxx"

def extract_text_pypdf2(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def llm_call(model, messages):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + OPENROUTER_API_KEY,  # Fixed to correct access to secrets
                "Content-Type": "application/json",
                "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app",  # To identify your app
                "X-Title": "lof-sims",
            },
            data=json.dumps({
                "model": model,
                "messages": messages,
            })
        )
    except requests.exceptions.RequestException as e:
        print(f"Error - make sure bills are paid!: {e}")
        return None
    # Extract the response content
    try:
        response_data = response
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.content}")
        return None
    return response_data.json()  # Adjusted to match expected JSON structure

pdf_path = '/home/abhinav_baiju/Documents/LoF/project4/lof-sims-interns/case.pdf'
case = extract_text_pypdf2(pdf_path)

start_time = time.time()
prompt = f"Extract relevant information for the following fields from the given case content: {case}\n\nCase Content:\nx"
response = llm_call("openai/gpt-4o", [{"role": "user", "content": prompt}])

# formatted_text = response.choices[0].message.content.replace("**","").replace("***","")
execution_time = round((time.time() - start_time),2)

request_cost = round(((response['usage']['prompt_tokens'] * 0.0050) / 1000),3)
response_cost = round(((response['usage']['completion_tokens'] * 0.0150) / 1000),3)
compute_cost = round(((execution_time * 0.0058) / 3600),6)

print("execution time: ",execution_time)

print('request cost: ',request_cost)
print('response cost: ',response_cost)
print('compute cost: ',compute_cost)


