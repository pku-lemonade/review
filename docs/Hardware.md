# Hardware

## Emerging Technologies

### Chiplets

#### Survey

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | Electronics | NUDT | Chiplet Heterogeneous Integration Technology—Status and Challenges | heterogeneous integration technology; interconnect interfaces and protocols; packaging technology|
| 2022 | CCF THPC | ICT | Survey on chiplets: interface, interconnect and integration methodology | development history; interfaces and protocols; packaging technology; EDA tool; standardization of chiplet technology |
| 2024 | IEEE CASS | Tsinghua University | Chiplet Heterogeneous Integration Technology—Status and Challenges | wafer-scale chip architecture; compiler tool chain; integration technology; wafer-scale system; fault tolerance |

### Novel Memory Technologies

#### NDP: 3D-stacked DRAM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | DAC | SNU | MoNDE: Mixture of Near-Data Experts for Large-Scale Sparse Models | NDP for MoE; activation movement; GPU-MoNDE load-balancing scheme |
| 2024 | ASPLOS | PKU | SpecPIM: Accelerating Speculative Inference on PIM-Enabled System via Architecture-Dataflow Co-Exploration | algorithmic and architectural heterogeneity; PIM resource allocation; multi-model collaboration workflow |

#### CIM: SRAM

#### CIM: RRAM

#### NVM

## Computer Architecture

### Domain-specific Accelerators

#### LLM Inference Accelerators

#### Graph Accelerators

### Memory Architecture

Refer to [Storage Systems](Software.md/#storage-systems).

#### Cache

#### Prefetcher

#### DRAM

### Communication Architecture

Refer to [Distributed Systems](Software.md/#distributed-systems).

#### Network-on-Chip

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | HPCA | George Washington University | Adapt-NoC: A Flexible Network-on-Chip Design for Heterogeneous Manycore Architectures | mux based adaptable router architecture; adaptable link design; reinforcement learning based subNoC optimization algorithm |
| 2022 | HPCA | Huawei | Application Defined On-chip Networks for Heterogeneous Chiplets: An Implementation Perspective | bufferless multi-ring NoC design; application-architecture-physical co-design method; architecture expressiveness; deadlock resolution SWAP mechanism |
| 2023 | ICCAD | University of Central Florida | ARIES: Accelerating Distributed Training in Chiplet-based Systems via Flexible Interconnects | directional bypassing link; ARIES link with transistor; ARIES all-reduce optimization algorithm |

#### Router

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2022 | HPCA | Chalmers University of Technology | FastTrackNoC: A NoC with FastTrack Router Datapaths | non-turning hops; direct FastTrack flit path; zero-load latency analysis |
| 2023 | HPCA | Tsinghua University | A Scalable Methodology for Designing Efficient Interconnection Network of Chiplets | interface grouping; hypercube construction algorithm; deadlock-free adaptive routing algorithm; safe/unsafe flow control; network interleaving method|

### Dataflow Architecture

Refer to [Heterogeneous Systems](Software.md/#heterogeneous-systems).

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | MICRO | Carnegie Mellon University | The TYR Dataflow Architecture: Improving Locality by Taming Parallelism | local tag spaces technique; space tag managing instruction set; CT based concurrent-block communication |
| 2024 | MICRO | UC Riverside | Sparsepipe: Sparse Inter-operator Dataflow Architecture with Cross-Iteration Reuse | producer-consumer reuse; cross-iteration reuse; sub-tensor dependency; OEI dataflow; sparsepipe architecture|

#### Data Mapping

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | HPCA | Georgia Tech | Heterogeneous Dataflow Accelerators for Multi-DNN Workloads | heterogeneous dataflow accelerators (HDAs) for DNN; dataflow flexibility; high utilization across the sub-accelerators |
| 2021 | HPCA | Georgia Institute of Technology | MAGMA: An Optimization Framework for Mapping Multiple DNNs on Multiple Accelerator Cores | sub-accelerator selection; fine-grained job prioritization; MANGA crossover genetic operators |
| 2023 | ISCA | Tsinghua University | MapZero: Mapping for Coarse-grained Reconfigurable Architectures with Reinforcement Learning and Monte-Carlo Tree Search | GAT based DFG and CGRA embedding; routing penalty based reinforcement learning; Monte-Carlo tree search space exploration |
| 2023 | VLSI | IIT Kharagpur | Application Mapping Onto Manycore Processor Architectures Using Active Search Framework | RNN based active search framework; IP-Core Numbering Scheme; active search with/without pretraining |
| 2023 | ISCA | Tsinghua University | Inter-layer Scheduling Space Definition and Exploration for Tiled Accelerators | inter-layer encoding method; temperal cut; spatial cut; RA tree analysis |
| 2024 | HPCA | Tsinghua University | Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators | layer-centric encoding method; DP-based graph partition algorithm; SA based D2D link communication optimization |

#### Task Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | TCAD | Arizona State University | Runtime Task Scheduling Using Imitation Learning for Heterogeneous Many-Core Systems | offline Oracle optimizaion strategy; hierarchical imitation learning based scheduling; two-level scheduling |
| 2023 | ICCAD | Peking University | Memory-aware Scheduling for Complex Wired Networks with Iterative Graph Optimization | topology-aware pruning algorithm; integer linear programming scheduling method; sub-graph fusion algorithm ; memory-aware graph partitioning|

### Reconfigurable Architecture

## Electronic Design Automation

## Performance Evaluation

### Modeling and Simulation

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | MICRO | Georgia Institute of Technology; NVIDIA | MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings | data-centric mapping; data reuse analysis; TemperalMap; SpatialMap; analytical cost model |
| 2022 | OSDI | UC Berkeley | Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning | inter-operator parallelisms; intra-operator parallelisms; ILP and DP hierarchical optimization |
| 2023 | MICRO | Peking University | TileFlow: A Framework for Modeling Fusion Dataflow via Tree-based Analysis | 3D design space of fusion dataflow; tree-based description; tile-centric notation |
| 2023 | ISPASS | Georgia Institute of Technology | ASTRA-sim2.0: Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale | graph-based training-loop execution; multi-dimensional heterogeneous topology construction; analytical network backend |
| 2024 | ISCA | Stanford University | The Dataflow Abstract Machine Simulator Framework | communicating sequential processes; event-queue free execution; context-channel based description; asynchronous distributed time |

### Performance Analysis

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2022 | ASPLOS | NC State University | ValueExpert: Exploring Value Patterns in GPU-accelerated Applications value-related inefficiencies | data value pattern recoginition; value flow graph; parallel intervals merging algorithm |
