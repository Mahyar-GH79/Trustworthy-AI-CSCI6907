# AI in Scientific Research

This workspace now supports a fully structured 30-prompt study for vision-language models, computer vision, and aerial-drone applications.

## Files

- `prompts.json`: the 30 prompts to run
- `run_openai_batch.py`: sends all prompts to the OpenAI Responses API
- `outputs/responses.json`: saved model outputs after running the script
- `results_log.csv`: score-only sheet for your 1-10 ratings
- `project_plan.md`: project framing and evaluation setup
- `blog_post_template.md`: final write-up template

## Run

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4.1"
python3 run_openai_batch.py
```

## Workflow

1. Run the batch script.
2. Open `outputs/responses.json`.
3. Read each saved response.
4. Enter your scores in `results_log.csv`.
5. Use the scored results for plots.

   
## Report
You can find the prompt templates and responses I used in the code here. You can also find the article [here](https://mahyarghazanfari.substack.com/p/c775ef0f-49d5-48d1-b565-622cd1dafaa6).

