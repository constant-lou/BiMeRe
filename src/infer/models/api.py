import requests
from utils.vl_utils import make_interleave_content

def load_model(model_name="GPT4", base_url="", api_key="", model="gpt-4-turbo-preview"):
    return {
        'model_name': model_name,
        'model': model,
        'base_url': base_url,
        'api_key': api_key
    }

def request_with_interleave_content(interleave_content, timeout=60, base_url="", api_key="", model="", model_name=None, verbose=False):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": interleave_content,
        "max_tokens": 2048
    }

    if verbose:
        print(f"[DEBUG] Request to {base_url}")
        print(f"[DEBUG] Model: {model}")
        print(f"[DEBUG] Messages sample: {interleave_content[:1]} ...")

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        response_json = response.json()

        return response_json["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e)
        print("Response text:", response.text)
        return None
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None
    except Exception as e:
        print("Unexpected error:", e)
        return None


def infer(prompts, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)
    verbose = kwargs.get('verbose', False)

    responses = []

    for data_id, prompt_set in enumerate(prompts):
        if verbose and data_id < 3:
            print(f"\n[DEBUG] Processing prompt #{data_id}")
            print("[DEBUG] Prompt set keys:", prompt_set.keys())

        messages = []

        if "conversations" in prompt_set:
            for idx, data in enumerate(prompt_set["conversations"]):
                user_payload = {"role": "user", "content": None}
                question = data["prompt"]

                images = data.get("images", [])
                images = ["<|image|>" + image if "<|image|>" not in image else image for image in images]

                interleave_content = make_interleave_content([question] + images)
                user_payload["content"] = interleave_content
                messages.append(user_payload)

                if idx != len(prompt_set["conversations"]) - 1 and "response" in data:
                    assistant_payload = {"role": "assistant", "content": data["response"]}
                    messages.append(assistant_payload)
        else:
            user_payload = {"role": "user", "content": None}
            question = prompt_set["prompt"]
            images = prompt_set.get("images", [])
            images = ["<|image|>" + image if "<|image|>" not in image else image for image in images]

            interleave_content = make_interleave_content([question] + images)
            user_payload["content"] = interleave_content
            messages.append(user_payload)

        response = request_with_interleave_content(messages, base_url=base_url, api_key=api_key, model=model, model_name=model_name, verbose=verbose)

        if response is not None:
            responses.append(response)

    return responses

