# Project Plan

## 1. Research Context

- Topic: AI in scientific research for vision-language models, computer vision, and aerial-drone applications
- My actual project or research area: Vision-language models, computer vision, and their applications in aerial drones
- Why this matters to me: I want to understand where GPT-like systems genuinely help my research workflow and where they become unreliable, shallow, or misleading.

## 2. Main Question

How useful and trustworthy is a GPT-like model for supporting this research task?

## 3. Specific Research Tasks To Test

Choose 2-4 tasks, for example:

- Hypothesis generation
- Literature summarization
- Experimental design
- Error analysis
- Writing or restructuring notes

My chosen tasks:

- Task 1: Hypothesis brainstorming
- Task 2: Literature summarization
- Task 3: Experimental recommendations

## 4. Evaluation Criteria

Use the same criteria every time. Score each from 1 to 10 in `results_log.csv`.

- Accuracy: Is the content factually correct?
- Helpfulness: Did it save time or improve the work?
- Specificity: Was it concrete enough to act on?
- Novelty: Did it generate ideas I would not have produced quickly myself?
- Trustworthiness: Did it cite, hedge, or reveal uncertainty appropriately?
- Creativity: Did it produce interesting and non-obvious ideas?
- Reliability: Would I feel comfortable building on this without major rewriting?
- Clarity: Was the answer organized and easy to interpret?
- Actionability: Could I directly turn the answer into a research step?
- Depth: Was the reasoning substantial rather than superficial?
- Failure modes: Hallucinations, shallow advice, wrong assumptions, bias, or overconfidence.

## 5. Comparison Baseline

Compare AI support against one of these:

- My normal workflow without AI
- My first-pass draft before AI help
- A smaller or weaker prompt

Chosen baseline:

- My normal workflow without GPT help
- My own first-pass ideas before prompting
- My manual literature search or experimental planning process

## 6. Data Collection Rules

- Use the 30 prepared prompts in `prompts.json`.
- Run them through the OpenAI API with `run_openai_batch.py`.
- Save all generated outputs in `outputs/responses.json`.
- Score every response in `results_log.csv`.
- Compare patterns across the three task groups, not just individual prompts.

## 7. Deliverables

- `outputs/responses.json` containing 30 model outputs
- Filled `results_log.csv` with 30 scored rows
- Final `blog_post_template.md`
