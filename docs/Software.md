# Software

## Computation and Language

### LLM Alignment

#### Self-Alignment

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | SJTU | Self-Alignment of Large Language Models via Monopolylogue-based Social Scene Simulation | social scene simulation; emulate realistic multiparty interactions and consequences; monopolylogue |

### LLM Finetune

#### Coding LLM Finetune

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | UMD | HPC-Coder-V2: Studying Code LLMs Across Low-Resource Parallel Languages | large synthetic parallel programming dataset; parallel code generation; HPC AI developer tools |

### LLM-Powered AI Agent

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | THU | LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination | hierarchical language agent; real-time human-AI coordination; slow mind & fast mind |

## Programming

### Compiler

## Operating Systems

### Virtualization

### Memory

#### Memory Partitioning

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | UMich | Mercury: QoS-Aware Tiered Memory System | contend for local memory; priority inversion; intra- and inter-tier interference; per-tier page reclaimation |

### Scheduling

#### General Task Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | NSDI | MIT | Shinjuku: Preemptive Scheduling for µsecond-scale Tail Latency | preemptive scheduling; single-address space OS; hardware-supported virtualization |
| 2021 | SOSP | UPenn | When Idling is Ideal: Optimizing Tail-Latency for Heavy-Tailed Datacenter Workloads with Perséphone | reserve cores; non-conserving; request dispatching algorithm |

#### Speculative Execution (Non-LLM)

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | MSR | Forerunner: Constraint-based Speculative Transaction Execution for Ethereum | constraint-based speculative transaction execution; many-future nature; specialized fast-path program |
| 2024 | Arxiv | Politecnico di Milano | Minimizing speculation overhead in a parallel recognizer for regular texts | speculation overhead; chunk automaton; reduced-interface DFA |

#### LLM-Related Scheduling

##### LLM Request Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | UCSB | Multi-Bin Batching for Increasing LLM Inference Throughput | binning-based scheduling strategy; queueing-theoretical analysis; asymptotical throughput optimality |
| 2024 | Arxiv | Yale | TimelyLLM: Segmented LLM Serving System for Time-sensitive Robotic Applications | segmented generation; time-sensitive scheduling; latency-guided batch size selection |

##### LLM Application-Level Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | SJTU | Parrot: Efficient Serving of LLM-based Applications with Semantic Variable | Semantic Variable; application-level information; LLM applications as first-class citizens |
| 2024 | OSDI | CUHK | Teola: Towards End-to-End Optimization of LLM-based Applications | mismatch between request-level scheduling and end-to-end  application performance; primitive-level dataflow graph; two-tier scheduling mechanism |
| 2024 | Arxiv | Yext | SLA Management in Reconfigurable Multi-Agent RAG: A Systems Approach to Question Answering | constantly changing and sometimes adverse conditions; Dynamically Reconfigurable Horizontal Scaling Framework; dynamically adjust resource allocation based on query requirements |

##### LLM Speculative Inference

Refer to non-LLM [speculative execution](#Speculative-Execution-(Non-LLM)).

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | F&M College | AMUSD: Asynchronous Multi-Device Speculative Decoding for LLM Acceleration | simultaneous and independent predictions; asynchronous speculative decoding; rollback mechanism |
| 2024 | Arxiv | Purdue University | Constrained Decoding with Speculative Lookaheads | computational expense of generating lookaheads; speculated lookaheads; task specific reward function |
| 2024 | Arxiv | Rutgers University | Interactive Speculative Planning: Enhance Agent Efficiency through Co-design of System and User Interface | active user intervention; speculative planning algorithm; UI-level rescheduling algorithm |
| 2024 | Arxiv | USTC | Parallel Speculative Decoding with Adaptive Draft Length | adaptive draft length; pre-verify and post-verify; draft-then-verify framework; mutual waiting problem |
| 2024 | Arxiv | SEU | SEED: Accelerating Reasoning Tree Construction via Scheduled Speculative Decoding | reasoning tree construction; parallel drafting with speculative decoding; FCFS queue verification |

#### DNN Scheduling

##### Task Offloading

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | USTC | Collaborative Inference for Large Models with Task Offloading and Early Exiting | early exit mechanism; jointly optimize its offloading strategy and the confidence threshold; distributed task offloading algorithm |

## Parallel Computing

### Storage Systems

### Distrbuted Systems

#### LLM Inference Systems

##### Surveys

###### System Optimization Surveys

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | Northeastern University | LLM Inference Serving: Survey of Recent Advances and Opportunities | KV cache and memory management; LLM computation optimization; Cloud LLM deployment; focus on system-level enhancements |
| 2024 | Arxiv | CSE Huawei | Software Performance Engineering (SPE) for Foundation Model-Powered Software (FMware) | performance concerns are often considered afterthoughts; continuous performance engineering; cognitive architecture design / communication protocols / tuning and optimization / deployment |
| 2024 | Arxiv | CUHK | A Survey on Inference Optimization Techniques for Mixture of Experts Models | model compression; expert skip; expert merge; sparse to dense; expert parallel; expert offloading |
| 2024 | Arxiv | PolyU | A Survey on Large Language Model Acceleration based on KV Cache Management | cache selection; budget allocation; cache merging; cache quantization; cache low-rank decomposition; attention grouping and sharing; memory management; hardware-aware design |

###### Application Surveys

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | PKU | Retrieval-Augmented Generation for AI-Generated Content: A Survey | Query Transformation; Data Augmentation; Recursive Retrieval; Chunk Optimization; Retriever Finetuning; Hybrid Retrieval; Re-ranking; Retrieval Transformation; Prompt Engineering; Decoding Tuning; Generator Finetuning; Output Rewrite; Adaptive Retrieval; Iterative RAG |
| 2024 | Arxiv | WHU | A survey on LLM-based multi-agent systems: workflow, infrastructure, and challenges | personalized characteristics; perceive environmental information; utilize memory mechanisms; mutual interaction; agent self-reflection |
| 2024 | Arxiv | PolyU | Deploying Foundation Model Powered Agent Services: A Survey | FM-powered agent services within the edge-cloud environment; low-level hardware perspective; high-level software perspective |

##### Long Sequence LLM Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | SJTU & Alibaba | Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache | inefficient model parallelism intra-instance; inefficient resource management inter-instance; KV cache scheduling |
| 2024 | Arxiv | SJTU | TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication | communication-oriented parallelism framework; inter-node P2P bidirectional communication bandwidth; optimization of attention block communication |

##### P-D Disaggregated Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | PKU | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | goodput-optimized; prefill-decoding interference；novel placement algorithm for p-d schema |
| 2024 | ISCA | University of Washington | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | optimized cache context transfer; performance per dollar; performance per watt; exploration of homogeneous and heterogeneous cluster deployments |
| 2024 | Arxiv | CMU | A System for Microserving of LLMs | fine-grained sub-request level actions; dynamic reconfiguration according to workloads; unified KV cache abstraction |

##### KV Cache Evict Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | NIPS | UT-Austin | H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | sparsity for small cache size; heavy-hitters; greedy algorithm for low-cost policy |
| 2024 | Arxiv | Fujitsu | CO2: Precise Attention Score Observation for improving KV Cache Replacement in Large Language Models | long measurement step; decay of the accumulated attention score; adjusting FIFO cache size |

##### KV Cache Reuse Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | UC Berkeley | BlendServe: Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching | offline batch inference; resource-aware prefix tree; compute-intensive / memory-intensive requests |
| 2024 | Arxiv | UChicago | CacheBlend: Fast Large Language Model Serving for RAG with  Cached Knowledge Fusion |  multiple precomputed text chunks; selective KV recompute; sparsity of attention matrices |
| 2024 | Arxiv | UChicago | DroidSpeak: Enhancing Cross-LLM Communication | selectively layer reuse; communication protocol for inter-agent exchanges; LLMs that share a common foundational model |
| 2024 | Arxiv | Microsoft | BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching | global prefix tree ahead-of-time; request reorder; horizontal fusioned prefix-shared attention kernel |
| 2024 | Arxiv | UC Berkeley | Optimizing LLM Queries in Relational Workloads | prefix sharing maximization; KV cache hit rate; deduplication and cost estimation techniques |

##### Fair Serving Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | Virginia Tech | Ensuring Fair LLM Serving Amid Diverse Applications | multi-tenant LLM platform; overload and interaction-driven throttling; weighted service counter |

#### Communication-Computation Overlap

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | NSDI | KAIST | ARK: GPU-driven Code Execution for Distributed Deep Learning | communication-motivated DL system; pipeline DMA engine; GPU-direct-controlled DMA |
| 2024 | ASPLOS | PKU | Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning | communication partition abstraction; hybrid LLM training tasks; 3-level decompose |
| 2024 | ASPLOS | UW–Madison | T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives | lightweight track and trigger; pre-programmed DMA commands; atomic memory update |
| 2024 | ASPLOS | UIUC | Two-Face: Combining Collective and One-Sided Communication for Efficient Distributed SpMM | distributed SpMM; sparsity-aware partition; Synchronous Stripes and Asynchronous Stripes |
| 2024 | Arxiv | AMD | Optimizing ML Concurrent Computation and Communication with GPU DMA Engines | concurrent computation and communication; compute and memory interference among concurrent kernels; schedule prioritization and careful resource partitioning |

### Heterogeneous Systems

#### General Applications

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2013 | SOSP | MSR Silicon Valley | Dandelion: a Compiler and Runtime for Heterogeneous  Systems | unified programming model; “single machine” abstraction; a rich object-oriented programming language for data-parallel computing |

#### Decentralized Training

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | ASPLOS | USC | Hop: Heterogeneity-aware Decentralized Training | iteration gap; queue-based synchronization; backup workers and bounded staleness |
| 2020 | ASPLOS | USC | Prague: High-Performance Heterogeneity-Aware Asynchronous Decentralized Training | Partial All-Reduce to reduce synchronization cost; group scheduling to avoid conflicts |

#### ML Training Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | SOSP | CMU | Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling | heterogeneity-aware and adaptivity-aware; ILP formulation for scheduling; bootstrapped from observing just a few mini-batches |

#### LLM Inference Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | CMU | Helix: Distributed Serving of Large Language Models via Max-Flow on Heterogeneous GPUs | LLM model placement as a max-flow problem; per-request pipeline; mixed integer linear programming |
| 2024 | Arxiv | UIC | Priority-Aware Model-Distributed Inference at Edge Networks | priority-aware model distributed inference algorithm; prioritization of ML inference tasks; model-distributed inferencing mechanism |
| 2024 | Arxiv | Yonsei University | Uncertainty-Aware Hybrid Inference with On-Device Small and Remote Large Language Models | hybrid language model; selectively skip uplink transmissions; uncertainty-aware |
| 2024 | Arxiv | UMD | Distributed Mixture-of-Agents for Edge Inference with Large Language Models | Mixture-of-Agents; semantics of the data being gossiped and its timeliness; queuing stability |

#### LLM Training Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | PKU | Demystifying Workload Imbalances in Large Transformer Model Training over Variable-length Sequences | data sampling imbalance; data packing imbalance; subgraph abstraction |
| 2024 | Arxiv | Ant Group | EDiT: A Local-SGD-Based Efficient Distributed Training Method for Large Language Models | Local Stochastic Gradient Descent (Local SGD); consistent stragglers within heterogeneous devices; hierarchical distribution strategy on a two-dimensional device mesh; layer by layer forward syncing; pseudo-gradient penalty method |
| 2024 | Arxiv | ZJU | Frenzy: A Memory-Aware Serverless LLM Training System for Heterogeneous GPU Clusters | efficient and low-overhead task-to-cluster scheduling; bin-packing algorithms; seamless and user-friendly |

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

##### Bottleneck Analysis

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2018 | PPoPP | Tsinghua University | vSensor: Leveraging Fixed-Workload Snippets of Programs for Performance Variance Detection | fixed-workload snippets; dependency propagation algorithm; lightweight on-line analysis algorithm |
| 2020 | SC | Tsinghua University | ScalAna: automating scaling loss detection with graph analysis | program structure graph; program performance graph; backtracking root cause detection algorithm |
| 2022 | PPoPP | Tsinghua University | Vapro: Performance Variance Detection and Diagnosis for Production-Run Parallel Applications | state transition graph; fixed workload snippets identification clustering algorithm; variance breakdown model; time of factors quantification method |
| 2024 | Arxiv | UGA | Performance Debugging through Microarchitectural Sensitivity and Causality Analysis | constraints propagation engine for causality analysis; differential analysis engine for sensitivity analysis |
| 2024 | SC | BUAA | GVARP: Detecting Performance Variance on Large-Scale Heterogeneous Systems | asynchronous state transition graph; parameter-based workload estimation method; asynchronous event tracing technology |

##### Variance Attribution

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2014 | ISPASS | Intel | A Top-Down Method for Performance Analysis and Counters Architecture | top-down bottleneck analysis method; frontend bound; bad speculation; retiring; backend bound |
| 2019 | SC | NC State University | Pinpointing Performance Inefficiencies via Lightweight Variance Profiling | function-level variance detection; stack based deep call chains maintain; on-the-fly binary analysis technique for calling context |

##### Benchmark

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2018 | ICPP | Washington University | Varbench: an Experimental Framework to Measure and Characterize Performance Variability | spatial/temperal variability; Resource Variability (RV) statistic |