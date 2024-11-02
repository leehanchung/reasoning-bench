import os
import asyncio
import time
import argparse
import csv
from typing import Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm


load_dotenv()
client = AsyncOpenAI()



async def call_openai_api(*, prompt: str, max_completion_token: int) -> Tuple[Any, float]:
    start_time = time.time()

    response = await client.chat.completions.create(
        model="o1-preview",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=max_completion_token
    )

    end_time = time.time()
    latency = end_time - start_time

    return response, latency


async def analyze_reasoning(prompt: str, answer: str, token_range: list, trials:int = 10) -> list:
    results = []

    tasks = [call_openai_api(prompt=prompt, max_completion_token=t) for t in token_range for _ in range(trials)]
    for t, (response, latency) in tqdm(
            zip([t for t in token_range for _ in range(trials)], await asyncio.gather(*tasks)),
            total=len(token_range),
            desc="analyzing reasoning"):

        results.append(
            {
                'max_tokens': t,
                'response': response.choices[0].message.content,
                'is_correct':  response.choices[0].message.content == answer,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'reasoning_tokens': response.usage.completion_tokens_details.reasoning_tokens,
                'total_tokens': response.usage.total_tokens,
                'latency': latency
             }
        )
    return results


def save_to_csv(results, filename):
    with open(filename, 'w', newline='') as f:
        fieldnames = ['max_tokens', 'response', 'is_correct', 'input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens', 'latency']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for r in results:
            writer.writerow(r)


async def main(max_completion_token: int) -> None:
    
    prompt = "what's larger? 9.11 or 9.8? answer only from 9.11 or 9.8. please think step by step"
    answer = "9.8"

    token_range = range(100, 5001, 100)

    results = await analyze_reasoning(prompt, answer, token_range)
    save_to_csv(results, 'results.csv')

    print('done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call OpenAI API and measure latency and token count")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens for the response")
    args = parser.parse_args()

    asyncio.run(main(args.max_tokens))

