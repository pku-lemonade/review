# Software

## Programming

### Compiler

#### CPU-GPU Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2013 | SOSP | MSR Silicon Valley | Dandelion: a Compiler and Runtime for Heterogeneous  Systems | unified programming model; “single machine” abstraction; a rich object-oriented programming language for data-parallel computing |

## Operating Systems

### Virtualization

### Scheduling

#### General Task Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | NSDI | MIT | Shinjuku: Preemptive Scheduling for µsecond-scale Tail Latency | preemptive scheduling; single-address space OS; hardware-supported virtualization |
| 2021 | SOSP | UPenn | When Idling is Ideal: Optimizing Tail-Latency for Heavy-Tailed Datacenter Workloads with Perséphone | reserve cores; non-conserving; request dispatching algorithm |

#### LLM-based Application Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | SJTU | Parrot: Efficient Serving of LLM-based Applications with Semantic Variable | Semantic Variable; application-level information; LLM applications as first-class citizens |

## Parallel Computing

### Storage Systems

### Distrbuted Systems

#### LLM Inference Systems

##### Long Sequence LLM Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | SJTU & Alibaba | Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache | inefficient model parallelism intra-instance; inefficient resource management inter-instance; KV cache scheduling |

##### P-D Disaggregated Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | PKU | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | goodput-optimized; prefill-decoding interference；novel placement algorithm for p-d schema |
| 2024 | ISCA | University of Washington | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | optimized cache context transfer; performance per dollar; performance per watt; exploration of homogeneous and heterogeneous cluster deployments;  |

##### KV Cache Reuse Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | UC Berkeley | BlendServe: Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching | offline batch inference; resource-aware prefix tree; compute-intensive / memory-intensive requests |
| 2024 | Arxiv | UChicago | CacheBlend: Fast Large Language Model Serving for RAG with  Cached Knowledge Fusion |  multiple precomputed text chunks; selective KV recompute; sparsity of attention matrices |
| 2024 | Arxiv | UChicago | DroidSpeak: Enhancing Cross-LLM Communication | selectively layer reuse; communication protocol for inter-agent exchanges; LLMs that share a common foundational model |
| 2024 | Arxiv | Microsoft | BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching | global prefix tree ahead-of-time; request reorder; horizontal fusioned prefix-shared attention kernel |

##### Fair Serving Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | Virginia Tech | Ensuring Fair LLM Serving Amid Diverse Applications | multi-tenant LLM platform; overload and
interaction-driven throttling; weighted service counter |

#### Communication-Computation Overlap

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | NSDI | KAIST | ARK: GPU-driven Code Execution for Distributed Deep Learning | communication-motivated DL system; pipeline DMA engine; GPU-direct-controlled DMA |
| 2024 | ASPLOS | PKU | Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning | communication partition abstraction; hybrid LLM training tasks; 3-level decompose |
| 2024 | ASPLOS | UW–Madison | T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives | lightweight track and trigger; pre-programmed DMA commands; atomic memory update |
| 2024 | ASPLOS | UIUC | Two-Face: Combining Collective and One-Sided Communication for Efficient Distributed SpMM | distributed SpMM; sparsity-aware partition; Synchronous Stripes and Asynchronous Stripes |

### Heterogeneous Systems

#### LLM Inference Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | CMU | Helix: Distributed Serving of Large Language Models via Max-Flow on Heterogeneous GPUs | LLM model placement as a max-flow problem; per-request pipeline; mixed integer linear programming |

## Performance Evaluation

### Modeling and Simulation

#### Performance Modeling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2009 | CACM | Berkeley | Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures | operational intensity; memory bound; compute bound |
| 2021 | Intelligent Computing | UC Berkeley | Hierarchical Roofline Performance Analysis for Deep Learning Applications | Nsight Compute based hierarchical roofline model; FP16、FP32 extension for ERT|

### Performance Analysis

#### Detection

Refer to [Compiler](#compiler) techniques.

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2014 | ISPASS | Intel | A Top-Down Method for Performance Analysis and Counters Architecture | top-down bottleneck analysis method; frontend bound; bad speculation; retiring; backend bound |
| 2018 | PPoPP | Tsinghua University | vSensor: Leveraging Fixed-Workload Snippets of Programs for Performance Variance Detection | fixed-workload snippets; dependency propagation algorithm; lightweight on-line analysis algorithm |
| 2019 | SC | NC State University | Pinpointing Performance Inefficiencies via Lightweight Variance Profiling | function-level variance detection; stack based deep call chains maintain; on-the-fly binary analysis technique for calling context |
| 2020 | SC | Tsinghua University | ScalAna: automating scaling loss detection with graph analysis | program structure graph; program performance graph; backtracking root cause detection algorithm |
| 2022 | PPoPP | Tsinghua University | Vapro: Performance Variance Detection and Diagnosis for Production-Run Parallel Applications | state transition graph; fixed workload snippets identification clustering algorithm; variance breakdown model; time of factors quantification method |
