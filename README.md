# Lemonade Review Guidelines

## About

The New Lemonade Review is a paper review list maintained by prospective and current PhD students @pku-lemonade/phds at [LEMONADE](https://www.youwei.xyz), Peking University. It is intended to provide a comprehensive overview of the quintessential research ideas in the field of computer architecture and systems.

> [!IMPORTANT]
> Always check the guidelines before submitting a PR.

## How to find the papers?

Choose your search strategy based on your research stage:

1. Broad Search:
    - Goal: Get a broad overview of the topic.
    - Method: Search in Google Scholar or read survey papers.
    - Pace: 10+ papers/week (mostly skimming).
1. Author-Focused Search
    - Goal: Understand a key researcher's approach (methods, presentation, evaluation).
    - Method: Search the author's papers in DBLP
    - Pace: 3-5 papers/week (skim first, then carefully read relevant papers)
1. Citation Chasing
    - Goal: Deep dive into a core paper (e.g., your intended main baseline).
    - Method: Follow its main baselines/references (backward look) and check papers that cite it (forward look).
    - Pace: 2-3 papers/week (requires careful reading). 
    - Note: Aim for at least 2 papers/week as you typically need to read the core paper alongside its main baseline. 

## What papers should be included?

1. Publication Type:
    - Include peer-reviewed full papers only.
    - Exclude workshop papers, posters, or abstracts.
    - Preprints are acceptable only for LLM-related topics.
1. Significant and Relevant papers
    - Prefer CCF Rank A papers.
    - During the Broad Search phase, ensure at least half the papers are either published in CCF Rank A venues, or have a relavance score over 3, as defined in the guidelines below).

## How to organize the papers?

1. Format:  
    1. Sort papers by year in each subsection.
    1. Use " " around titles or tags containing "," (important for CSV)
1. Authors
    1. Use the **affiliation** of the **corresponding author** (or **last author**).
    1. Use standard, globally recognized abbreviations for affiliations. 
1. Subsections
    1. Put papers only in **Level 3** or **Level 4** subsections.
    1. Group papers by a common challenge in each subsection. Briefly state that challenge at the beginning of the subsection. 
    1. **Min 2 papers** per subsection. If only one fits, find a partner paper, or connect this paper to a related subsection.
    1. **Max 5 papers** per subsection. If more fit,  split the subsection. 
1. Tags
    1. Use specific tags for key techniques or contributions, such as "xxx algorithm", "yyy model", "zzz architecture", etc.
    1. Don't use broad area tags ("performance" or "architecture").
    1. Don't use vague feature tags ("expresiveness", "scalability", "efficiency")unless combined with a specific technique (e.g., "xxx algorithm for scalability").
    1. Explain potentially unclear acronyms ("XYZ framework") in tags.
1. Links
    1. Link related subsections (e.g., link hardware papers to related software papers sharing an idea, link compiler papers to related system papers sharing a technique).

## Review Scores (How to read the papers?)

### Presentation

**Quick check: Abstract, Introduction, Figures.**

- 5: Definitely stealing some ideas for how they explained things or made their figures. I wish my papers looked this good.
- 4: Maybe a figure or explanation I'll remember and use later.
- 3: I can write as good as it.
- 2: Kind of a pain to read, or the figures are confusing/ugly. Hard to tell what's going on easily.

### Evaluation

**Check the end: Experiments, Results, Reproducibility.**

- 5: Solid real-world tests. NVDA could actually use or bet on.
- 4: Solid tests with real hardware. But the setup is not real-wrold ready.
- 3: Okay tests with open source simulators. The results aren't obviously wrong.
- 2: Okay tests with close source simulators. The results aren't obviously wrong.
- 1: The results just don't make sense.

### Novelty

**Compare with related papers.**

- 5: Totally new idea.
- 4: New take on old ideas, or combines them smartly.
- 3: An old idea applied to a totally new area/topic.
- 2: An old idea applied to a similar problem.

### Significace/Relevance

Note: Don't put the score in the review record because this is subjective to your work

- 5: Everyone need to read this paper
- 4: I will put this paper on my side when I write my next paper
- 3: I will cite this paper in my next paper
- 2: I will cite this paper in my next survey paper
