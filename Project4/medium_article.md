# I Tried Using GPT as a Research Assistant for Vision-Language Models on Drones

I wanted to test something simple: if I used GPT in a real research workflow, where would it actually help, and where would it start sounding smarter than it really was?

My topic was the intersection of vision-language models, computer vision, and aerial drones. Instead of asking GPT random one-off questions, I built a small structured evaluation around 30 prompts and grouped them into three kinds of research work:

- brainstorming hypotheses
- summarizing literature
- making experimental recommendations

I then scored the outputs across usefulness, novelty, clarity, reliability, trustworthiness, and a few other dimensions. I also tracked risk-related scores like overconfidence and bias risk.

This post is a plain-English write-up of what I learned.

![Figure 1: Main summary figure](figures/icml_main_figure.png)

Put this image right after the intro. It gives readers the big picture before the examples.

## The Setup

I wrote 30 prompts total:

- 10 for hypothesis brainstorming
- 10 for literature summarization
- 10 for experimental recommendations

All of them were about research problems around aerial drones, remote sensing, computer vision, and vision-language systems.

Then I ran them through GPT, saved the responses, and manually scored each one.

The results were pretty clear: GPT was strongest when I asked it to help structure research work. It was weaker when I asked it to act like a literature expert without giving it sources.

## What Worked Best

The strongest category by far was experimental recommendations.

That was not too surprising in hindsight. GPT is very good at turning a fuzzy research direction into a checklist, protocol, or experiment plan. It tends to do well when the task is:

- structured
- procedural
- close to standard scientific workflow
- easy to break into variables, metrics, baselines, and ablations

One of the best responses was the dataset audit prompt. GPT turned that into a practical pretraining checklist with things like:

- defining the audit scope first
- checking label quality
- checking geographic bias
- checking duplicate imagery
- checking train-test leakage
- writing down minimum acceptable quality thresholds

That answer was not especially novel, but it was very useful. It felt like the kind of thing that could save real time if I were about to start a project and wanted to avoid avoidable mistakes.

Another strong one was the efficiency-versus-quality experiment prompt. GPT proposed a hardware-in-the-loop setup with power budget, memory budget, thermal limits, model-size comparisons, and a Pareto frontier between quality and onboard cost. That is exactly the kind of structure that helps when moving from “this sounds like a cool idea” to “here is a real experiment I can run.”

![Figure 2: Metric comparison across task types](figures/task_metric_barplot.png)

Put this image after the “What Worked Best” section. It helps show that experimental recommendations scored highest on the most practical metrics.

## Hypothesis Brainstorming Was Good, but You Still Have to Filter Hard

The hypothesis prompts were often strong on creativity and direction-setting. GPT was good at taking a theme and turning it into several testable angles instead of giving vague motivational text.

For example, when I asked about adverse weather and aerial vision-language models, one of the hypotheses it proposed was that fog would hurt small-object language grounding more than coarse scene classification. That is actually a pretty solid research hypothesis. It came with a mechanism, suggested variables to manipulate, and metrics to measure.

That is useful because it moves you quickly from:

"weather is probably bad for models"

to something closer to:

"fog may disproportionately hurt object-level grounding compared to scene-level understanding, especially at higher altitude and smaller object scales"

That is much more testable.

The same thing happened in the safety-critical prompts. GPT was strong at surfacing plausible failure modes, like a drone system sounding confident while hallucinating disaster severity or victim presence from weak visual evidence.

But this category also had one of the biggest limitations: some of the hypotheses sounded good partly because they were written well. A polished hypothesis is not the same thing as a strong one. A few entries drifted into “interesting sounding” more than “carefully grounded.”

So my takeaway here is:

GPT is good at generating candidate hypotheses, but I still need to act as the filter. It is a brainstorming partner, not a research advisor.

## Literature Summarization Was the Weakest Category

This was the clearest lesson in the whole project.

GPT was decent at giving broad summaries of a field. It could organize an answer, break topics into themes, and identify common challenges. But when I looked more carefully, literature summarization was the least trustworthy use case of the three.

The biggest issue was not that the answers were obviously terrible. The issue was that they were often too smooth.

A good example was the datasets-and-benchmarks prompt. Instead of staying tightly focused on aerial drones and related vision-language settings, the answer drifted into generic computer vision datasets like ImageNet, CIFAR, MNIST, and COCO. None of those mentions are inherently wrong, but the answer started to feel like a standard ML overview rather than a focused summary of the actual subfield I cared about.

That kind of drift matters. If I were moving fast, I could easily mistake a broad, polished answer for a targeted literature review.

This is where GPT felt most like a dangerous intern: helpful at producing a first pass, but risky if I stop checking.

![Figure 3: Quality-risk frontier](figures/quality_risk_frontier.png)

Put this image after the literature section. It helps show the general pattern that some answers had decent quality but were not equally trustworthy.

## The Best Use of GPT Was Not “Knowing More.” It Was Organizing Research Work

This was probably the most important thing I learned.

GPT did not impress me most when it tried to be the authority on the field.

It impressed me most when it helped with the shape of research:

- turning vague ideas into testable hypotheses
- turning goals into evaluation protocols
- turning project ideas into scoped experiments
- turning “I should probably audit this dataset” into a step-by-step checklist

That is a different role from “tell me the truth about the literature.”

In other words, GPT was more useful as a workflow accelerator than as a source of scientific authority.

## What I Would Trust It With, and What I Would Not

After going through all 30 outputs, here is my honest version.

I would trust GPT to help with:

- hypothesis generation
- experiment scaffolding
- checklists
- ablation plans
- evaluation protocols
- project scoping
- error analysis templates

I would not trust GPT on its own for:

- literature reviews that need precise citation-level accuracy
- identifying the exact right datasets or papers without verification
- making claims about what the field “already knows”
- safety-critical reasoning if the output is not grounded in evidence

That difference matters a lot. One use case saves time. The other can quietly distort your understanding.

## The Most Blog-Worthy Result for Me

The most interesting result was that the highest-scoring outputs were not the flashiest ones.

The winners were the ones that behaved like strong research infrastructure:

- dataset audit procedure
- safety-critical evaluation framework
- efficiency-versus-quality experiment design
- error analysis workflow

Those are not dramatic. But they are exactly the kind of thing that makes research better.

Meanwhile, the weaker outputs were often the ones that sounded broad and complete but were less tightly grounded in the exact problem.

That feels like a pretty useful lesson for working with LLMs in science:

the real value is often in structure, not in authority.

![Figure 4: Top-scoring prompts](figures/icml_top_prompts.png)

Put this image near the end, right before the conclusion. It gives readers a concrete sense of which prompt types were most effective.

## Final Takeaway

Going into this, I expected GPT to be most useful for summarizing papers and generating high-level ideas.

After actually testing it, I came away with a more specific view.

GPT was best when I asked it to help me organize the mechanics of research. It was weaker when I treated it like a reliable map of the literature.

So if I use it again in my own workflow, I will use it as:

- a research planner
- a hypothesis generator
- a protocol designer
- a structured first-draft assistant

But not as a final source of truth.

That is a much narrower role than the hype suggests, but honestly, it is still a very useful one.

## Image Placement Summary

If you want the cleanest Medium layout, use the images in this order:

1. `figures/icml_main_figure.png`
   Put after the intro and setup.
2. `figures/task_metric_barplot.png`
   Put after the section on what worked best.
3. `figures/quality_risk_frontier.png`
   Put after the literature summarization section.
4. `figures/icml_top_prompts.png`
   Put near the end, before the final takeaway.
