
---

# README for Project3

# Project3 — LLM Bias Experiment

This project tests whether an LLM responds differently to the same scientific question when the user is described with different personas.

## Goal

The experiment compares responses across personas such as:

- student
- parent
- scientist
- journalist
- policymaker
- skeptic
- anxious user
- no-context user

It uses questions about:

- GMO safety
- nuclear energy safety
- vaccines and autism
- climate change consensus

## Metrics

For each response, the project measures:

- word count
- hedge density
- deflection rate
- fact coverage
- sentiment
- average sentence length

## Requirements

Install dependencies:

```bash
pip install openai matplotlib seaborn pandas numpy scipy
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Run:

```bash
python main.py
```

All results are saved in:

```bash
results/
```
## Generated files include:

metrics_raw.csv — raw response-level data

metrics_summary.csv — aggregated summary

metrics_table.tex — LaTeX table

heatmap_facts.png

barplot_words.png

barplot_hedges.png

barplot_deflect.png

radar_chart.png

scatter_words_vs_facts.png

boxplot_words.png

Experiment settings

## The script currently uses:

model: gpt-4o-mini

8 personas

4 questions

3 runs per persona-question pair


## Final Results
Some of the final results are shown here. For the complete set of outputs based on your preference, run the experiments with your own parameters, and analyzee the
outputs.

![Fig1](barplot_deflect.png)
![Fig2](barplot_hedges.png)
![Fig3](barplot_words.png)
![Fig4](boxplot_words.png)
![Fig5](scatter_words_vs_facts.png)
![Fig6](radar_chart.png)
