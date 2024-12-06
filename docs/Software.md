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

## Parallel Computing

### Storage Systems

### Distrbuted Systems

#### LLM Inference Systems

##### P-D Disaggregated Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | PKU | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | goodput-optimized; prefill-decoding interference；novel placement algorithm for p-d schema |
| 2024 | ISCA | University of Washington | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | optimized cache context transfer; performance per dollar; performance per watt; exploration of homogeneous and heterogeneous cluster deployments;  |

#### Communication-Computation Overlap

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | NSDI | KAIST | ARK: GPU-driven Code Execution for Distributed Deep Learning | communication-motivated DL system; pipeline DMA engine; GPU-direct-controlled DMA |
| 2024 | ASPLOS | PKU | Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning | communication partition abstraction; hybrid LLM training tasks; 3-level decompose |
| 2024 | ASPLOS | UW–Madison | T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives | lightweight track and trigger; pre-programmed DMA commands; atomic memory update |
| 2024 | ASPLOS | UIUC | Two-Face: Combining Collective and One-Sided Communication for Efficient Distributed SpMM | distributed SpMM; sparsity-aware partition; Synchronous Stripes and Asynchronous Stripes |

### Heterogeneous Systems

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
