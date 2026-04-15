# Prompt Log

This project no longer uses a manual prompt notebook.

- Prompt definitions live in `prompts.json`.
- Model outputs are saved automatically to `outputs/responses.json`.
- Your scoring sheet is `results_log.csv`.

Run the batch script to populate model outputs:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4.1"
python3 run_openai_batch.py
```
