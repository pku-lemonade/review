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

#### Hardware Description Language

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | ASPLOS | Cornell | A compiler infrastructure for accelerator generators | Solid: 3, Novelty: 3, Presentation: 4; a split representation combining a high-level control flow language with a hardware-like structural language; pass-based compiler; systolic array generator; live-range-based register-sharing |

## Operating Systems

### Virtualization

### Memory

#### Memory Partitioning

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | UMich | Mercury: QoS-Aware Tiered Memory System | contend for local memory; priority inversion; intra- and inter-tier interference; per-tier page reclaimation |

#### LLM Memory Management

##### General LLM Memory Management

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | THU | Jenga: Effective Memory Management for Serving LLM with Heterogeneity | Solid: 4, Novelty: 3, Presentation: 4; fixed-size embeddings; full-prefix dependency; two-level memory allocator |

##### LLM Quantization Methods

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | UVa | HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference | method without dequantization; homomorphic quantization method for matrix multiplication; requantization elimination |
| 2025 | Arxiv | PKU | Bitnet.cpp: Efficient Edge Inference for Ternary LLMs | ternary mpGEMM library; avoid intricate bit-level manipulations; achieving lossless inference for BitNet b1.58 |
| 2025 | Arxiv | SJTU | MILLION: Mastering Long-Context LLM Inference Via Outlier-Immunized KV Product Quantization | Solid: 4, Novelty: 2, Presentation: 3; a non-uniform quantization algorithm based on product quantization; leverages sparse computation and asynchronous quantization; distributes quantization power unevenly across channels |

##### KV Cache Storage Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | NVIDIA | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | block-sparse format; customizable attention template; dynamic load-balanced scheduling framework |
| 2025 | Arxiv | PKU | FairKV: Balancing Per-Head KV Cache for Fast Multi-GPU Inference | imbalanced KV cache compression mitigation; fair-copying for load balancing; best-effort assignment |

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

##### Systems with Other Caches (Not / Not just KV Cache)

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | KAIST | Efficient LLM Inference with Activation Checkpointing and Hybrid Caching | activation checkpointing; KV-activation hybrid caching; balanced approach to determine the best ratio |

### Scheduling

#### General Task Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | NSDI | MIT | Shinjuku: Preemptive Scheduling for µsecond-scale Tail Latency | preemptive scheduling; single-address space OS; hardware-supported virtualization |
| 2021 | SOSP | UPenn | When Idling is Ideal: Optimizing Tail-Latency for Heavy-Tailed Datacenter Workloads with Perséphone | reserve cores; non-conserving; request dispatching algorithm |

#### Heterogeneous Device Task Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | NUS | Data-aware Dynamic Execution of Irregular Workloads on Heterogeneous Systems | lightweight and input-aware framework; multiobjective and multi-constraint design space; dynamically creating optimal schedules |
| 2025 | Arxiv | Georgia Tech | HARP: A Taxonomy for Heterogeneous and Hierarchical Processors for Mixed-reuse Workloads | a taxonomy to classify the heterogeneous and hierarchical accelerators; characterize hardware organization of different accelerators; classify based on relative location of sub-accelerators |

#### Speculative Execution (Non-LLM)

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | MSR | Forerunner: Constraint-based Speculative Transaction Execution for Ethereum | constraint-based speculative transaction execution; many-future nature; specialized fast-path program |
| 2024 | Arxiv | Politecnico di Milano | Minimizing speculation overhead in a parallel recognizer for regular texts | speculation overhead; chunk automaton; reduced-interface DFA |

#### LLM-Related Scheduling

##### Cloud Datacenter LLM Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | Azure | TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms | thermal/power property characterization; dynamically adjust in response to power or cooling failures; thermal- and poweraware manner |

##### LLM Request Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | UCSB | Multi-Bin Batching for Increasing LLM Inference Throughput | binning-based scheduling strategy; queueing-theoretical analysis; asymptotical throughput optimality |
| 2024 | Arxiv | Yale | TimelyLLM: Segmented LLM Serving System for Time-sensitive Robotic Applications | segmented generation; time-sensitive scheduling; latency-guided batch size selection |
| 2025 | Arxiv | MSRI | Niyama : Breaking the Silos of LLM Inference Serving | Solid: 4, Novelty: 3, Presentation: 4; QoS-driven LLM inference serving system; co-scheduling requests with diverse QoS targets on a shared rather than siloed infrastructure; allows graceful service degradation during overload conditions; deadline slack; a hybrid prioritization and an eager relegation policy |

##### LLM Application-Level Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | SJTU | Parrot: Efficient Serving of LLM-based Applications with Semantic Variable | Semantic Variable; application-level information; LLM applications as first-class citizens |
| 2024 | OSDI | CUHK | Teola: Towards End-to-End Optimization of LLM-based Applications | mismatch between request-level scheduling and end-to-end  application performance; primitive-level dataflow graph; two-tier scheduling mechanism |
| 2024 | Arxiv | Yext | SLA Management in Reconfigurable Multi-Agent RAG: A Systems Approach to Question Answering | constantly changing and sometimes adverse conditions; Dynamically Reconfigurable Horizontal Scaling Framework; dynamically adjust resource allocation based on query requirements |
| 2025 | Arxiv | UC Berkeley | Autellix: An Efficient Serving Engine for LLM Agents as General Programs | formalize agentic programs as dynamic, non-deterministic DAGs; non-clairvoyant scheduler; simple load-balancing policy to balance data locality and KV-cache recomputation |
| 2025 | ICDCS | SJTU | LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications | Solid: 4, Novelty: 3, Presentation: 4; a DAG with regular stage, LLM stage, dynamic stage; bayesian network-based profiler; identify uncertainty-reducing stages |

##### LLM Speculative Inference

Refer to non-LLM [speculative execution](#Speculative-Execution-(Non-LLM)).

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | F&M College | AMUSD: Asynchronous Multi-Device Speculative Decoding for LLM Acceleration | simultaneous and independent predictions; asynchronous speculative decoding; rollback mechanism |
| 2024 | Arxiv | Purdue University | Constrained Decoding with Speculative Lookaheads | computational expense of generating lookaheads; speculated lookaheads; task specific reward function |
| 2024 | Arxiv | Rutgers University | Interactive Speculative Planning: Enhance Agent Efficiency through Co-design of System and User Interface | active user intervention; speculative planning algorithm; UI-level rescheduling algorithm |
| 2024 | Arxiv | USTC | Parallel Speculative Decoding with Adaptive Draft Length | adaptive draft length; pre-verify and post-verify; draft-then-verify framework; mutual waiting problem |
| 2024 | Arxiv | SEU | SEED: Accelerating Reasoning Tree Construction via Scheduled Speculative Decoding | reasoning tree construction; parallel drafting with speculative decoding; FCFS queue verification |

###### Spec + Others

| 2025 | Arxiv | Huawei | Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling | speculative MoE; speculative token shuffling; speculative expert pre-grouping |
| 2025 | INFOCOM | UoA | SPIN: Accelerating Large Language Model Inference with Heterogeneous Speculative Models | internal neurons sparsification; model-agnostic acceleration framework; dynamic early-exit thresholds; multi-layered feature fusion |

##### LLM Serving Outages and Incidents

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | arXiv | Vrije Universiteit Amsterdam | An Empirical Characterization of Outages and Incidents in Public Services for Large Language Models | empirical characterization of outages; failure recovery optimization; public LLM service reliability |

##### Energy-Optimized LLM Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | arXiv | UvA | GREEN-CODE: Optimizing Energy Efficiency in Large Language Models for Code Generation | dynamic early exit; energy-aware code generation; reinforcement learning for llms |

#### DNN Scheduling

##### Task Offloading

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | USTC | Collaborative Inference for Large Models with Task Offloading and Early Exiting | early exit mechanism; jointly optimize its offloading strategy and the confidence threshold; distributed task offloading algorithm |

## Parallel Computing

### Storage Systems

#### SSD Management
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | EuroSys | Samsung | Towards Efficient Flash Caches with Emerging NVMe Flexible Data Placement SSDs | "NVMe Flexible Data Placement (FDP) SSDs" for data segregation; targeted data placement for reduced device write amplification; FDP-enabled CacheLib architecture; theoretical DLWA model for CacheLib |
| 2025 | arXiv | SDU | Managing Hybrid Solid-State Drives Using Large Language Models | "LLM-based auto-tuning framework" for hybrid SSD management; hybrid SSD parameter categorization; performance-sensitive parameter selection; prompt engineering for LLM integration; dynamic configuration optimization in hybrid SSDs |

### Distrbuted Systems

#### Remote Memory
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | TC | Georgia Tech | Hierarchical Orchestration of Disaggregated Memory | "XMemPod architecture" for hierarchical memory orchestration; "compressed swap page table (CSPT)" for metadata management; hybrid swap-out algorithm for memory utilization; proactive swap-in optimization for performance; "RDMA-based remote memory sharing" for low-latency access |

#### Scratchpad Memory
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | ASPLOS | Cornell University | Beyond Static Parallel Loops: Supporting Dynamic Task Parallelism on Manycore Architectures with Software-Managed Scratchpad Memories | Solid: 3, Novelty: 3, Presentation: 3; work-stealing based dynamic task parallelism; stack/task queue in SPM; read-only data duplication |

#### I/O Characterization and Optimization

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | UOregon | Parallel I/O Characterization and Optimization on Large-Scale HPC Systems: A 360-Degree Survey | different HPC I/O stack layers; profiling and tracing tools; tuning echniques |

#### LLM Training Systems

##### General Optimizations

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | THU | Enhancing Memory Efficiency in Large Language Model Training Through Chronos-aware Pipeline Parallelism | chronos-aware pipeline parallelism; temporal locality optimization; activation balancing |
| 2025 | Arxiv | NUS | PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization | selective offload strategy; memory offload optimization; pipeline parallelism scalability; lifespan-based offloading |
| 2025 | Arxiv | UCSD | WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training | Solid: 5, Novelty: 2, Presentation: 4; workload-aware variable-length document packing; per-document sharding strategy; adaptive sharding selection mechanism; delay execution of extremely long documents |
| 2025 | EuroSys | University of Toronto | Mist: Efficient Distributed Training of Large Language Models via Memory-Parallelism Co-Optimization | Solid: 4, Novelty: 2, Presentation: 4; fine-grained overlap-centric scheduling; symbolic-based performance analysis; imbalance-aware hierarchical tuning |

##### Optimizations on Special Scene

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | ArXiv | HKU | Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism | Fully Sharded Sparse Data Parallelism (FSSDP); sparsely materializes MoE parameters; two sparse collective communications |

##### Experiments

| 2025 | Arxiv | JSC | Memory and Bandwidth are All You Need for Fully Sharded Data Parallel | Solid: 4, Novelty: 1, Presentation: 2; an extensive analysis of the FSDP training distribution strategy; a grid search methodology; both simulation and empirical results |

##### Multi-Modal Optimizations

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | ByteDance | OrchMLLM: Orchestrate Multimodal Data with Batch Post-Balancing to Accelerate Multimodal Large Language Model Training | Solid: 4, Novelty: 3, Presentation: 4; multimodal mini-batch imbalance; batch post-balancing algorithm; node-wise all-to-all communicator for practical rearrangement of mini-batches |

##### Kernel-Level Optimizations

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | HUST | CFP: Low-overhead Profiling-based Intra-operator Parallelism Generation by Preserving Communication-Free Structures | Solid: 5, Novelty: 3, Presentation: 4; model segment profile-based cost model; communication-free tensor partition propagation property; extracting a set of unique model segments; Communication-Free Preserve |

#### Many-Core Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2018 | SC | Intel | Many-Core Graph Workload Analysis | multicore simulator sniper; selective caching and prefetching; heterogeneous high-performance low-power cores |
| 2018 | DATE | UGA | Parallel Code Generation of Synchronous Programs for a Many-core Architecture | banked memory mapping; worst-case response time analysis |
| 2025 | IPDPS | The University of Chicago | Optimizing Fine-Grained Parallelism Through Dynamic Load Balancing on Multi-Socket Many-Core Systems | lock-less and concurrent task queue xqueue; distributed tree barrier; NUMA-aware redirect push/work stealing |

##### Fault Propagation

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2008 | ASPLOS | UIUC | Understanding the Propagation of Hard Errors to Software and Implications for Resilient System Design | stuck-at fault; bridging fault; software failure detection |
| 2010 | PRDC | UBC | Modeling the Propagation of Intermittent Hardware Faults in Programs | instruction based intermittent fault; dynamic dependency graph(DDG) based propagation modeling |
| 2015 | SC | IBM | Understanding the Propagation of Transient Errors in HPC Applications | fault propagation in MPI application; fault classification:V,ONA,WO,PEX,C; fault propagation speed factors |
| 2023 | ISCA | University of Chicago | Understanding and Mitigating Hardware Failures in Deep Learning Training Accelerator Systems | NVDLA based fault injection framework; re-execution based light-weight recovery technique; failure effects:SlowDegrade,SharpSlowDegrade,SharpDegrade,LowTestAccuracy |

##### Fault Injection Technique

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2008 | VLSI | DISCA | Enhancement of Fault Injection Techniques Based on the Modification of VHDL Code | saboteurs and mutants technique based fault injection; VHDL level fault-tolerance mechanism | 
| 2014 | DSN | UBC | Quantifying the Accuracy of High-Level Fault Injection Techniques for Hardware Faults | fault injection quantification; assembly level fault injection; LLVM compiler based fault injector |

##### Communication

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | UCLM | Understanding intra-node communication in HPC systems and Datacenters | intra- and inter-node simulation model; intra-node network interface bottleneck; impacts of communication pattern |

#### LLM Inference Systems

##### Communication-Focused Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | Apple | SPD: Sync-Point Drop for efficient tensor parallelism of Large Language Models | sync-point drop; block-wise sensitivity analysis; attention output synchronization reduction |

##### SLO-Aware Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | ArXiv | UC Berkeley | AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding | fine-grained speculative decoding; token tree verification; slo customization |
| 2025 | ArXiv | UIUC | HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location | online-offline request co-location; interference-aware profiler; latency predictor; adaptive scheduler |
| 2025 | Arxiv | PKU | Memory Offloading for Large Language Model Inference with Latency SLO Guarantees | effectively captures the tension between meeting SLOs and maximizing host memory usage; dynamic offloading interval; per-bus coordinator |
| 2025 | Arxiv | Huawei | Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization | hybrid offline-online scheduling; preemptive scheduling for hardware utilization; lagrangian method for cost efficiency evaluation |
| 2025 | FAST | THU | Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot | Solid: 4, Novelty: 2, Presentation: 3; PD-disaggregate system; kv-cache centered; global kv-cache pool; dynamic SLO scheduler; paged KV-Cache storage |

##### Surveys

###### System Optimization Surveys

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | Northeastern University | LLM Inference Serving: Survey of Recent Advances and Opportunities | KV cache and memory management; LLM computation optimization; Cloud LLM deployment; focus on system-level enhancements |
| 2024 | Arxiv | CUHK | A Survey on Inference Optimization Techniques for Mixture of Experts Models | model compression; expert skip; expert merge; sparse to dense; expert parallel; expert offloading |
| 2024 | Arxiv | PolyU | A Survey on Large Language Model Acceleration based on KV Cache Management | cache selection; budget allocation; cache merging; cache quantization; cache low-rank decomposition; attention grouping and sharing; memory management; hardware-aware design |
| 2025 | Arxiv | THU | Beyond A Single AI Cluster: A Survey of Decentralized LLM Training | resource-driven paradigm; community-driven decentralization; organizational decentralization; decentralized LLM training taxonomy |
| 2025 | Arxiv | FIU | Distributed LLMs and Multimodal Large Language Models: A Survey on Advances, Challenges, and Future Directions | Solid: (Survey), Novelty: (Survey), Presentation: 2; distributed solutions for LMs; workload imbalance in LLM training; M-ICL; model security enhancement |

###### Application Surveys

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | PKU | Retrieval-Augmented Generation for AI-Generated Content: A Survey | Query Transformation; Data Augmentation; Recursive Retrieval; Chunk Optimization; Retriever Finetuning; Hybrid Retrieval; Re-ranking; Retrieval Transformation; Prompt Engineering; Decoding Tuning; Generator Finetuning; Output Rewrite; Adaptive Retrieval; Iterative RAG |
| 2024 | Arxiv | WHU | A survey on LLM-based multi-agent systems: workflow, infrastructure, and challenges | personalized characteristics; perceive environmental information; utilize memory mechanisms; mutual interaction; agent self-reflection |
| 2024 | Arxiv | PolyU | Deploying Foundation Model Powered Agent Services: A Survey | FM-powered agent services within the edge-cloud environment; low-level hardware perspective; high-level software perspective |

##### Multimodal Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | UW–Madison | LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models | query-block distributed exchange; shared visual token recomputation; sequence-parallelism with minimal communication overhead |
| 2025 | Arxiv | Microsoft | Towards Efficient Large Multimodal Model Serving | fine-grained stage-aware resource management; multimodal workload-specific scheduling; model architecture-specific optimizations |
| 2025 | Arxiv | Huawei | Efficiently Serving Large Multimedia Models Using EPD Disaggregation | encode-prefill-decode disaggregation; multimodal cache; intra-request parallel |
| 2025 | Arxiv | TU/e | Fine-tuning Multimodal Transformers on Edge: A Parallel Split Learning Approach | Multimodal Parallel Split Learning; computation-efficient training; server-side loss aggregation mechanism |
| 2025 | Arxiv | HUST | FastCache: Optimizing Multimodal LLM Serving through Lightweight KV-Cache Compression Framework | resource-aware KV-cache memory pool; multimodal KV-cache compression; modality-specific compression |

##### Mixture-of-Experts LLM Systems

###### Expert Offloading and Placement

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | DATE | UC Berkeley | DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference | data-aware offloading; predictive pre-calculation; sequence-specific expert allocation |
| 2025 | Arxiv | Stevens Tech | fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving | expert map; iteration-level probability distributions; track fine-grained input semantic embeddings; semantic-based and trajectorybased |
| 2025 | Arxiv | Georgia Tech | MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing | ILP for expert placement; cross-layer dependencies; minimizing total dispatched token number |
| 2025 | EuroMLSys | EPFL | Accelerating MoE Model Inference with Expert Sharding | expert sharding for load balancing; tensor sharding for moe experts; fused expert computations for reduced kernel launches |
| 2025 | DAC | PKU | HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference | Solid: 4, Novelty: 2, Presentation: 3; dynamically balances workloads across GPUs and CPUs; impact-driven prefetching; MoE-specialized cache management |

###### Batching and Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | Alibaba | Static Batching of Irregular Workloads on GPUs: Framework and Application to Efficient MoE Model Inference | statically batching irregular workloads; batch-task-tile partition; decompress the mapping and dispatch the workload |
| 2025 | Arxiv | University of Edinburgh | MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching | module-based batching; high-throughput MoE inference; full KV-cache offloading |
| 2025 | Arxiv | KTH | Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference | fine-grained preemption; priority-aware scheduling; per-expert queues; expert-level preemption |

###### Memory and Communication Efficiency

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | ByteDance | Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts | fine-grained communication-computation overlapping for efficient MoE execution; dependency resolving method; adaptive workload assignment method; shared data buffers between communication and computation operations |
| 2025 | Arxiv | University of Virginia | eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference | expert prediction; task-aware expert loading; task-aware request scheduling |

###### Architectural Innovations

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | Shanghai AI | Linear-MoE: Linear Sequence Modeling Meets Mixture-of-Experts | linear sequence modeling with MoE; sparse activation via moe layers; hybrid models combining linear-moe and transformer-moe layers |
| 2025 | Arxiv | UC Berkeley | HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs | Solid: 5, Novelty: 3, Presentation: 4; zebra parallelism; attention-expert disaggregation; asymmetric expert assignment mechanism; gather and squeeze strategy |

###### Compute-Kernel-Level Optimizations

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | SJTU | Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores | dual-side structured sparsity; sparse-sparse matrix multiplication kernel; vector-wise + 2:4 hybrid sparsity; token-aware activation compression |

##### Long Sequence LLM Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | SJTU & Alibaba | Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache | inefficient model parallelism intra-instance; inefficient resource management inter-instance; KV cache scheduling |
| 2024 | Arxiv | SJTU | TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication | communication-oriented parallelism framework; inter-node P2P bidirectional communication bandwidth; optimization of attention block communication |
| 2025 | Arxiv | CWRU | Longer Attention Span: Increasing Transformer Context Length with Sparse Graph Processing Techniques | sparse attention with graph computing perspective; work-optimal graph algorithms; achieve "true sparsity" |
| 2025 | MLSys | MIT | LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention | unified sparse attention; hybrid static and dynamic sparsity; hierarchical kv cache management with query-centric pruning |
| 2025 | Arxiv | PKU | ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs | hybrid data parallelism; data-aware sharding; a heuristic algorithm that reorganizes data assignment based on the characteristics of data and pipeline parallelism |

##### P-D Disaggregated Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | OSDI | PKU | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | goodput-optimized; prefill-decoding interference；novel placement algorithm for p-d schema |
| 2024 | ISCA | University of Washington | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | optimized cache context transfer; performance per dollar; performance per watt; exploration of homogeneous and heterogeneous cluster deployments |
| 2024 | Arxiv | CMU | A System for Microserving of LLMs | fine-grained sub-request level actions; dynamic reconfiguration according to workloads; unified KV cache abstraction |
| 2025 | Arxiv | PKU | ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments | two-level hierarchical optimization; tabu search algorithm for GPU partition; a lightweight re-scheduling mechanism |

###### P-D Disaggregated System Optimizations

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | ByteDance | KVDirect: Distributed Disaggregated LLM Inference | tensor-centric communication mechanism; pull-based KV cache transfer; dynamic GPU resource scheduling via RDMA |
| 2025 | Arxiv | SYSU | Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation | Solid: 4, Novelty: 3, Presentation: 4; attention disaggregation and offloading mechanism; low-latency decoding synchronization; resource-efficient prefill colocation; load-aware offloading scheduling |
| 2025 | Arxiv | Alibaba | FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling | Solid: 4, Novelty: 2, Presentation: 4; analyze the communication patterns; KV cache structure adjustment method; load-aware scheduling |
| 2025 | Arxiv | Huawei | Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation | Solid: 3, Novelty: 2, Presentation: 3; attention disaggregation and offloading; load-aware offloading cchedule; resource-efficient prefill colocation |

##### Throughput-Optimized Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | HKUST | Improving the End-to-End Efficiency of Offline Inference for Multi-LLM Applications Based on Sampling and Simulation | Solid: 4, Novelty: 3, Presentation: 4; sampling-then-simulation cost model; model-level pipeline parallelism; minimumtotal-latency application scheduling |

##### Fair Serving Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | Virginia Tech | Ensuring Fair LLM Serving Amid Diverse Applications | multi-tenant LLM platform; overload and interaction-driven throttling; weighted service counter |
| 2025 | Arxiv | UIUC | Hierarchical Autoscaling for Large Language Model Serving with Chiron | hierarchical backpressure; interactive requests and batch requests; mixed instances |
| 2025 | Arxiv | UC Berkeley | Locality-aware Fair Scheduling in LLM Serving | deficit-based longest prefix matching; distributed deficit-round coordination; prefix-aware fairness bound analysis |

##### Prefix Cache

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | Nips  | Stanford  | SGLang: Efficient Execution of Structured Language Model Programs | Solid: 4, Novelty: 3, Presentation: 4; KV-Cache share; python-like DSL; compute graph; LRU cache management stragety |
| 2024 | arXiv | Microsoft | ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition | Solid: 4, Novelty: 2, Presentation: 3 prefix aware attention compute; manage kv-cache chunks as prefix tree; reduce kv-cache redundancy |

#### Communication-Computation Overlap

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | NSDI | KAIST | ARK: GPU-driven Code Execution for Distributed Deep Learning | communication-motivated DL system; pipeline DMA engine; GPU-direct-controlled DMA |
| 2024 | ASPLOS | PKU | Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning | communication partition abstraction; hybrid LLM training tasks; 3-level decompose |
| 2024 | ASPLOS | UW–Madison | T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives | lightweight track and trigger; pre-programmed DMA commands; atomic memory update |
| 2024 | ASPLOS | UIUC | Two-Face: Combining Collective and One-Sided Communication for Efficient Distributed SpMM | distributed SpMM; sparsity-aware partition; Synchronous Stripes and Asynchronous Stripes |
| 2024 | Arxiv | AMD | Optimizing ML Concurrent Computation and Communication with GPU DMA Engines | concurrent computation and communication; compute and memory interference among concurrent kernels; schedule prioritization and careful resource partitioning |

#### Tensor Execution Optimization

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | OSDI  | PKU     | Mirage: A Multi-Level Superoptimizer for Tensor Programs | Solid: 4, Novelty: 3, Presentation: 4; auto algebraically transfer tensor; using DAG to search configuration space; auto generate kernel function |

### Heterogeneous Systems

#### General Applications

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2013 | SOSP | MSR Silicon Valley | Dandelion: a Compiler and Runtime for Heterogeneous  Systems | unified programming model; “single machine” abstraction; a rich object-oriented programming language for data-parallel computing |
| 2025 | EuroSys | SJTU | Improving GPU Sharing Performance through Adaptive Bubbleless Spatial-Temporal Sharing | Solid 3; Novelty 2; Presentation 4; "Bubble-less" spatial-temporal sharing; kernel squad scheduling; fine-grained concurrent kernel management |

#### Decentralized Serving

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | ASPLOS | USC | Hop: Heterogeneity-aware Decentralized Training | iteration gap; queue-based synchronization; backup workers and bounded staleness |
| 2020 | ASPLOS | USC | Prague: High-Performance Heterogeneity-Aware Asynchronous Decentralized Training | Partial All-Reduce to reduce synchronization cost; group scheduling to avoid conflicts |
| 2025 | Arxiv | UC Berkeley | DeServe: Towards Affordable Offline LLM Inference via Decentralization |  decentralized LLM inference; high-latency optimization; idle GPU utilization; modular on-chain integration |
| 2025 | Arxiv | HKUST | DreamDDP: Accelerating Data Parallel Distributed LLM Training with Layer-wise Scheduled Partial Synchronization | partial synchronization based local SGD; DFS algorithm with pruned search space; enables the opportunity of overlapping communication and computation |

#### ML Training Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | SOSP | CMU | Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling | heterogeneity-aware and adaptivity-aware; ILP formulation for scheduling; bootstrapped from observing just a few mini-batches |

#### LLM Inference Systems

##### Mobile & Edge-Network Serving

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | UIC | Priority-Aware Model-Distributed Inference at Edge Networks | priority-aware model distributed inference algorithm; prioritization of ML inference tasks; model-distributed inferencing mechanism |
| 2024 | Arxiv | Yonsei University | Uncertainty-Aware Hybrid Inference with On-Device Small and Remote Large Language Models | hybrid language model; selectively skip uplink transmissions; uncertainty-aware |
| 2024 | Arxiv | UMD | Distributed Mixture-of-Agents for Edge Inference with Large Language Models | Mixture-of-Agents; semantics of the data being gossiped and its timeliness; queuing stability |
| 2025 | Arxiv | PKU | SplitLLM: Hierarchical Split Learning for Large Language Model over Wireless Network | hierarchical split learning; edge-cloud collaboration; LoRA adapter update |
| 2025 | Arxiv | SJTU | HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators | both layer-level and tensor-level GPU-NPU parallelism; different tensor partition strategies; fast synchronization mechanism based on predictable kernel waiting times; tensor partition solver |

##### Large Heterogeneous System Serving 

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | CMU | Helix: Distributed Serving of Large Language Models via Max-Flow on Heterogeneous GPUs | LLM model placement as a max-flow problem; per-request pipeline; mixed integer linear programming |
| 2025 | Arxiv | USTC | PICE: A Semantic-Driven Progressive Inference System for LLM Serving in Cloud-Edge Networks | progressive inference paradigm; ensemble learning mechanism; semantic-level parallel data processing |
| 2025 | ICLR | HKUST | HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment | a combination of graph partitioning and max-flow algorithm; TP and PP with disaggregation; bottleneck and underutilized edges; swap edges |
| 2025 | arXiv | CMU | Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures | Solid 3; Novelty 2; Presentation 2; SKIP profiling tool; TKLQT metric for CPU/GPU boundedness; proximity score kernel fusion |

#### LLM Training Systems

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | PKU | Demystifying Workload Imbalances in Large Transformer Model Training over Variable-length Sequences | data sampling imbalance; data packing imbalance; subgraph abstraction |
| 2024 | Arxiv | Ant Group | EDiT: A Local-SGD-Based Efficient Distributed Training Method for Large Language Models | Local Stochastic Gradient Descent (Local SGD); consistent stragglers within heterogeneous devices; hierarchical distribution strategy on a two-dimensional device mesh; layer by layer forward syncing; pseudo-gradient penalty method |
| 2024 | Arxiv | ZJU | Frenzy: A Memory-Aware Serverless LLM Training System for Heterogeneous GPU Clusters | efficient and low-overhead task-to-cluster scheduling; bin-packing algorithms; seamless and user-friendly |
| 2025 | Arxiv | OSU | Scaling Large Language Model Training on Frontier with Low-Bandwidth Partitioning | low-bandwidth interconnects; three-level hierarchical partitioning strategy; improved hierarchical partitioning on top of ZeRO++ |
| 2025 | Arxiv | PKU | Split Fine-Tuning for Large Language Models in Wireless Networks | split fine-tuning; device and server partition; novel compression scheme and resource management algorithm |
| 2025 | Arxiv | Neuchatel | SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks | partial pipeline parallelism; stage skipping; path scheduling algorithm |

## Performance Evaluation

### Modeling and Simulation

#### Performance Modeling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2009 | CACM | Berkeley | Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures | operational intensity; memory bound; compute bound |
| 2014 | | ETH Zurich | Extending the Roofline Model: Bottleneck Analysis with Microarchitectural Constraints | Solid: 4, Novelty: 3, Presentation: 3; dag-based performance model; Tomasulo’s greedy algorithm; scheduled dag based bottleneck modeling |
| 2021 | Intelligent Computing | UC Berkeley | Hierarchical Roofline Performance Analysis for Deep Learning Applications | Nsight Compute based hierarchical roofline model; FP16、FP32 extension for ERT|
| 2025 | Arxiv | Google | Concorde: Fast and Accurate CPU Performance Modeling with Compositional Analytical-ML Fusion | Solid: 2, Novelty: 2, Presentation: 3; per-resource throughput analysis; fine-grained performance attribution |

#### LLM Serving

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | arXiv | KAIST | LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale | iteration-level simulation; computation reuse optimization; heterogeneous accelerator mapping |

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

##### Bottleneck Optimization

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Arxiv | Indian Institute of Science | Performance Characterization and Optimizations of Traditional ML Applications | dummy datasets generation; software-based prefetching for neighbour/tree-based workloads; data layout and computation re-ordering algorithm |

##### Variance Attribution

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2014 | ISPASS | Intel | A Top-Down Method for Performance Analysis and Counters Architecture | top-down bottleneck analysis method; frontend bound; bad speculation; retiring; backend bound |
| 2016 | TPDS | ICT | Understanding Big Data Analytics Workloads on Modern Processors | top-down analysis for big data workload; pipeline-characteristics basd performance implication analysis; BigDataBench benchmark |
| 2019 | SC | NC State University | Pinpointing Performance Inefficiencies via Lightweight Variance Profiling | function-level variance detection; stack based deep call chains maintain; on-the-fly binary analysis technique for calling context |

##### Benchmark

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2018 | ICPP | Washington University | Varbench: an Experimental Framework to Measure and Characterize Performance Variability | spatial/temperal variability; Resource Variability (RV) statistic |
| 2021 | IEEE Access | D-ITET | DAMOV: A New Methodology and Benchmark Suite for Evaluating Data Movement Bottlenecks | NDP focused workload characterization methodology; memory-bound function identification; locality-based clustering; memory bottlenecks classification |
