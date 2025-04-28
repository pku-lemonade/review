# Lemonade Review Guidelines

## About

The New Lemonade Review is a paper review list maintained by prospective and current PhD students @pku-lemonade/phds at [LEMONADE](https://www.youwei.xyz), Peking University. It is intended to provide a comprehensive overview of the quintessential research ideas in the field of computer architecture and systems.

## How to find the papers?

> [!IMPORTANT]
> Choose one of the first three strategies as your focus **each week**, depending on your current research needs.

> [!IMPORTANT]
> Check arXiv **every day**, as an ongoing habit!

### **Broad Search**

- Goal: Get a broad overview of the topic.
- Method: Search in Google Scholar or read survey papers.
- Pace: 5+ papers/week (**mostly skimming**).

### **Author-Focused Search**

- Goal: Understand a key researcher's approach (methods, presentation, evaluation).
- Method: Search the author's papers in DBLP.
- Pace: 2-5 papers/week (**skim first, then carefully read relevant papers**).

### **Citation Chasing**

- Goal: Deep dive into a core paper (e.g., your intended main baseline).
- Method: Follow its main baselines/references (backward look) and check papers that cite it (forward look).
- Pace: 2-3 papers/week (**requires careful reading**).
- **[!NOTE]** Aim for at least 2 papers/week as you typically need to read the core paper alongside its main baseline.

### **arXiv Monitoring**

- Goal: Stay updated with the latest research developments
- Method: Check:
  - [arXiv.DC](https://papers.cool/arxiv/cs.DC)
  - [arXiv.ET](https://papers.cool/arxiv/cs.ET)
  - [arXiv.AR](https://papers.cool/arxiv/cs.AR)
  - [arXiv.PF](https://papers.cool/arxiv/cs.PF)
- Pace: 0+ papers/week (**mostly skimming**).

## What papers should be included?

- Peer-reviewed full papers ONLY.
- NO workshop papers, posters, or abstracts.
- Preprints are okay only for LLM-related topics.

> [!IMPORTANT]
> Ensure that **at least half** the papers are either published in CCF Rank A venues, or have a relavance score at least 3.

## How to organize the papers?

### Format

- Sort papers by year in each subsection.
- CSV: Use quotes (" ") around titles and tags containing commas ","
- PR: Include search strategy, keywords, and note which papers, if any, were read carefully (as opposed to skimmed) in PR title and description.

### Authors

- Use the **affiliation** of the **corresponding author** (or **last author**).
- Use standard, globally recognized abbreviations for affiliations.

### Subsections

- Put papers only in **Level 3** or **Level 4** subsections.
- Group papers by a **common challenge** in each subsection. Briefly state that challenge at the beginning of the subsection.
- **Min 2 papers** per subsection. If only one fits, find a partner paper, or connect this paper to a related subsection.
- **Max 5 papers** per subsection. If more fit,  split the subsection.

### Tags

- Use specific tags for key techniques or contributions, such as "xxx algorithm", "yyy model", "zzz architecture", etc.
- Don't use broad area tags ("performance" or "architecture").
- Don't use vague feature tags ("expresiveness", "scalability", "efficiency")unless combined with a specific technique (e.g., "xxx algorithm for scalability").
- Explain potentially unclear acronyms ("XYZ framework") in tags.

### Links

- Link related subsections (e.g., link hardware papers to related software papers sharing an idea, link compiler papers to related system papers sharing a technique).

## How to read the papers? (Review Scores)

### Presentation

> [!Tip]
> **Skimming**: Title, Abstract, Introduction, Figures.

- 5: Definitely stealing some ideas for how they explained things or made their figures. I wish my papers looked this good.
- 4: I will take note a figure or explanation and use in my next paper.
- 3: I can write as good as it.
- 2: Kind of a pain to read, or the figures are confusing/ugly. Hard to tell what's going on easily.

### Evaluation

> [!Tip]
> **Skimming**: Experiments, Results, Reproducibility.

- 5: Solid real-world tests. NVDA could actually use or bet on.
- 4: Solid tests with real hardware. But the setup is not real-wrold ready.
- 3: Okay tests with open source simulators. The results aren't obviously wrong.
- 2: Okay tests with close source simulators. The results aren't obviously wrong.
- 1: The results just don't make sense.

### Novelty

> [!Note]
> **Requires careful reading**: Assessing novelty means comparing the paper critically to related work. If you only skimmed the paper, either skip this score or assign at most 2.

- 5: Totally new idea; groundbreaking.
- 4: New take on existing ideas, or combines them smartly.
- 3: An existing idea applied to a new area/topic.
- 2: An existing idea applied to the same problem with standard incremental contributions.
- 1: An existing idea applied to the same problem with minor variations.

### Relevance

> [!NOTE]
> Don't put the score in the review record because this score should be personal.

- 5: Everyone need to read this paper
- 4: I will put this paper on my side when I write my next paper
- 3: I will cite this paper in my next paper
- 2: I will cite this paper in my next survey paper
