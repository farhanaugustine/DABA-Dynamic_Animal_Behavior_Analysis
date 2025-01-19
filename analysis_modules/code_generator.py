import asyncio
import logging
import httpx
import json
import yaml
import time
from typing import AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return {}

async def _query_ollama_stream(prompt, model, ollama_url, timeout=600, max_retries=3, retry_delay=2) -> AsyncGenerator[str, None]:
    """Queries an Ollama model and yields the response stream with retries."""
    url = f"{ollama_url}/api/chat"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                logging.info(f"Querying Ollama model '{model}', attempt {attempt + 1}...")
                async with client.stream("POST", url, json=data) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        try:
                            chunk_str = chunk.decode('utf-8')
                            if chunk_str.strip():
                                json_data = json.loads(chunk_str)
                                if "message" in json_data and "content" in json_data["message"]:
                                    yield json_data["message"]["content"]
                        except json.JSONDecodeError:
                            logging.warning(f"JSON decode error in chunk, attempt {attempt + 1}: {chunk_str[:100]}...")
                            continue
                return  # If successful, exit the retry loop
        except httpx.TimeoutException as e:
            logging.warning(f"Timeout error querying Ollama, attempt {attempt + 1}: {e}")
        except httpx.HTTPError as e:
            logging.warning(f"HTTP Error querying Ollama, attempt {attempt + 1}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error querying Ollama, attempt {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

    logging.error(f"Failed to query Ollama after {max_retries} attempts.")
    yield "No code generated"

async def generate_analysis_code(user_prompt, ollama_url, roi_file_path) -> AsyncGenerator[str, None]:
    """Generates Python code using an LLM."""
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Config file not found: config.yaml")
        yield "Error: Config file not found"
        return
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        yield f"Error: {e}"
        return

    code_gen_model = config.get("code_gen_model", "codegemma:latest") # Default to codegemma:latest if not specified
    if not code_gen_model:
        logging.warning("Warning: 'code_gen_model' not found in config.yaml. Using default 'codegemma:latest'.")
        code_gen_model = "codegemma:latest"

    prompt = f"""
    You are an expert Python programmer specializing in animal behavior analysis.
    Given the following user request: "{user_prompt}",
    and the following path to a JSON file containing ROI definitions: "{roi_file_path}",
    Generate a Python script that performs the requested analysis.
    The script should load the ROI definitions from the JSON file. Or allow the user to input their own ROIs as [(x,y),(x,y),(x,y),(x,y)].
    The script should load the data from a CSV file.
    The script should output the results to the console.
    Return ONLY the code, without any surrounding text.
    """

    async for chunk in _query_ollama_stream(prompt, code_gen_model, ollama_url):
        yield chunk

if __name__ == '__main__':
    async def main():
        user_prompt = "Calculate the average speed of the body center in the open arms and closed arms."
        config = load_config()
        ollama_url = config.get("ollama_url")
        roi_file_path = "rois.json"

        if not ollama_url:
            print("Error: 'ollama_url' not found in config.yaml.")
            return

        async for chunk in generate_analysis_code(user_prompt, ollama_url, roi_file_path):
            print(chunk, end="", flush=True)
        print()

    asyncio.run(main())