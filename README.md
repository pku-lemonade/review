# About NLR

The New Lemonade Review is a paper review list maintained by [LEMONADE](https://www.youwei.xyz) at Peking University. The list is updated regularly by prospective and current PhD students. It is intended to provide a comprehensive overview of the quintessential research ideas in the field of computer architecture and systems.

## Guidelines

> [!IMPORTANT]
> Always check the guidelines before submitting a PR.

### What papers should be included?

1. Do not add workshop papers, posters, or abstracts.
2. Add peer-reviewed full papers. Add preprints when they are important and have not been published in a conference or journal.

### How to organize the papers?

1. Sort the papers by year in descending order in the same subsection.
1. Authors
    1. Use the **affiliation** of the **corresponding/last author**.
    1. If the affiliation name is **longer than 3 words**, use abbreviations to avoid too many new lines in a table cell. For example, always use abbreviations like "HKUST"，"Georgia Tech". 
1. Tags
    1. Do not use general tags like "performance" or "architecture" that describe areas or topics.
    1. Use specific tags that describe the techniques or contributions of the paper, such as "xxx algorithm", "yyy model", "zzz architecture", etc.
    1. Do not use "features" as tags, such as "expresiveness", "scalability", "efficiency", etc. However, it is appropriate to use "xxx algorithm for scalability" as a tag to highlight the motivation of each technique.
    1. If the tags contain weired acronyms, use double quotes and explain them.
1. Links
    1. Section and its subsections are implicitly linked.
    1. Do not create links if the subsections are linked by transitivity.
    1. Create links between subsections that are related to each other. For example, always try to find out links from the hardware list to the software list.
1. You are free to modify the subsections at all levels by adherring to the following rules
    1. All papers should be placed in subsections at level 4 or level 5.
    1. There should be **at least 2 papers** in the same subsection. If there is only one paper suitable for the subsection, find another paper, or connect the subsection with a link from or to another paper.
    1. There should be **no more than 5 papers** in the same subsection. If more than 5 papers are suitable for the subsection, create a new subsection, or mark the most recent papers as delete.
