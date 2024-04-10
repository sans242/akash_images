import pandas as pd
import asyncio
import openai
from aiohttp import ClientSession
from tqdm.asyncio import tqdm as async_tqdm
import nest_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai.error import OpenAIError
import csv
import os
import pickle
import numpy as np
from tqdm import tqdm


# Apply nest_asyncio to allow nested use of asyncio in the notebook
nest_asyncio.apply()


# Create a holder for ClientSession
class SessionHolder:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = ClientSession()
        openai.aiosession.set(self.session)
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()


# A method to initialize the session
async def initialize_session():
    SessionHolder.session = ClientSession()
    openai.aiosession.set(SessionHolder.session)

# A method to close the session
async def close_session():
    if SessionHolder.session:
        await SessionHolder.session.close()

# Set API key and aiohttp session for OpenAI
openai.api_type = "azure"
openai.api_base = input("Enter OpenAI API base: ")
openai.api_version = "2023-12-01-preview"
openai.api_key = input("Enter OpenAI API key: ")
engine_name = input("Enter engine name: ")
temperature = float(input("Enter temperature value: "))

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=1, max=120), retry=(retry_if_exception_type(OpenAIError)))
async def chat_completion(content):
    data_id, data_name, prompt, temp_id1 = content
    chat_completion_resp = await openai.ChatCompletion.acreate(
        engine=engine_name,
        messages=[{"role":"system","content":"You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown."},{"role": "user", "content": prompt}],
        temperature=temperature,
		max_tokens=2000,
        timeout=120
    )
    try:
        answer = chat_completion_resp['choices'][0]['message'].get('content', '')
    except IndexError:
        answer = ''
    completion_tokens = chat_completion_resp['usage']['completion_tokens']
    prompt_tokens = chat_completion_resp['usage']['prompt_tokens']
    total_tokens = chat_completion_resp['usage']['total_tokens']
    success = True
    error = "successful"
    cost_est = float(((prompt_tokens * 0.0015 * 83) / 1000) + ((completion_tokens * 0.002 * 83) / 1000))
    data = [data_id, data_name, prompt,temp_id1, answer, success, error, completion_tokens, prompt_tokens, total_tokens, cost_est, temperature, engine_name]
    with open(f'{output_dir}/{temp_id1}.plk', 'wb') as file:
        pickle.dump(data, file)
    return data

async def create_chat_completions(df, id_var, name_var, prompt_var, temp_id):
    tasks = []
    for data_name, data_id, prompt, temp_id1 in zip(df[name_var], df[id_var], df[prompt_var], df[temp_id]):
        content = data_id, data_name, prompt, temp_id1
        task = asyncio.create_task(chat_completion(content))
        tasks.append(task)

    responses = []
    pending_content = []
    
    while tasks:
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        for task in done:
            try:
                result = await task
                responses.append(result)
            except Exception as e:
                if isinstance(e, openai.error.OpenAIError):
                    pending_content.append(task)
                else:
                    print(f"Unexpected error occurred: {str(e)}")

    return responses, [t for t in pending_content if not t.done()]


async def process_chat_completions(filepath, id_var, name_var, prompt_var, output_dir):
    # Determine the file type
    _, file_extension = os.path.splitext(filepath)

    # Load the file into a DataFrame
    if file_extension == '.csv':
        df = pd.read_csv(filepath)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError('File type not recognized. Please provide a CSV or Excel file.')
    
    #df = df.head(100)
    df['temp_id'] = 0
    #df[cost] = 0
    for i in range(len(df)):
        df['temp_id'][i] = i

    all_responses = []
    all_pending_content = []

    chunks = np.array_split(df, len(df) // 750 + 1)

    async with SessionHolder() as session:
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}")
            responses, pending_content = await create_chat_completions(chunk, id_var, name_var, prompt_var, temp_id)
            all_responses.extend(responses)
            if pending_content:
                print("Retrying pending content for current chunk")
                pending_df = pd.DataFrame(pending_content, columns=[id_var, name_var, prompt_var, temp_id])
                responses, _ = await create_chat_completions(pending_df, id_var, name_var, prompt_var, temp_id)
                all_responses.extend(responses)
            await asyncio.sleep(5)  # Delay for 1 second
    return df, all_responses, all_pending_content


filepath = input("Enter the path of the input file: ")
id_var = input("Enter the variable for id: ")
name_var = input("Enter the variable for name: ")
prompt_var = input("Enter the variable for prompt: ")
output_dir = input("Enter the path of the output directory: ")
output_csv = input("Enter the path of the output CSV file: ")
temp_id = 'temp_id'
#cost = 'cost'
# Run async function with all DataFrame elements and get responses
loop = asyncio.get_event_loop()
df, all_responses, all_pending_content = loop.run_until_complete(process_chat_completions(filepath, id_var, name_var, prompt_var, output_dir))

# Print the number of total data, total responses, and total pending tasks
print("Total Data: ", len(df))
print("Total Responses: ", len(all_responses))
print("Total Pending Content: ", len(df)-len(all_responses))

def pickle_to_csv():
    # Initialize an empty list to store dictionaries
    #output_dir = input("Enter the directory containing the pickle files: ")
    #output_csv = input("Enter the path of the output CSV file: ")
    data_list = []
    # Get a list of all pickle files in output_dir
    pickle_files = [f for f in os.listdir(output_dir) if f.endswith('.plk')]
    # Load data from each pickle file and append it to the list
    files_processed = 0
    for file_name in tqdm(pickle_files, desc="Processing pickle files", unit="file"):
        with open(os.path.join(output_dir, file_name), 'rb') as file:
            data = pickle.load(file)
            data_list.append(data)
            files_processed += 1
    # Explicitly set column names
    columns = ["data_id", "data_name", "prompt","temp_id1", "answer", "success",
               "error", "completion_tokens", "prompt_tokens", "total_tokens",
               "cost_est", "temperature","engine_name"]
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list, columns=columns)
    # Export the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
pickle_to_csv()

