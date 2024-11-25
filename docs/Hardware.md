# Hardware

## Emerging Technologies

### Chiplets

#### Survey

| Year | Venue | Authors | Title | Tags |
|-|-|-|------------------|--------------|
| 2020 | Electronics | National University of Defense Technology | Chiplet Heterogeneous Integration Technology—Status and Challenges | heterogeneous integration technology; chiplet concept |
| 2022 | CCFTHPC | Institute of Computing Technology, Chinese Academy of Sciences | Survey on chiplets: interface, interconnect and integration methodology | development history; standardization of chiplet technology |
| 2024 | IEEE Circuits and Systems Magazine | Tsinghua University | Chiplet Heterogeneous Integration Technology—Status and Challenges | wafer-scale computing |

### Novel Memory Technologies

#### NDP: 3D-stacked DRAM

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
|-|-|-|------------------|--------------|
| 2021 | HPCA | George Washington University | Adapt-NoC: A Flexible Network-on-Chip Design for Heterogeneous Manycore Architectures | reinforcement learning; application-aware; subnoc | 
| 2022 | HPCA | Huawei | Application Defined On-chip Networks for Heterogeneous Chiplets: An Implementation Perspective | non-buffer structure; multi cycle |
| 2023 | ICCAD | University of Central Florida | ARIES: Accelerating Distributed Training in Chiplet-based Systems via Flexible Interconnects | reconfigurable interconnects; all-reduce algorithm |

#### Router

| Year | Venue | Authors | Title | Tags |
|-|-|-|------------------|--------------|
| 2022 | HPCA | Chalmers University of Technology | FastTrackNoC: A NoC with FastTrack Router Datapaths | non-turning hops; direct path |
| 2023 | HPCA | Tsinghua University | A Scalable Methodology for Designing Efficient Interconnection Network of Chiplets | interface grouping; hypercube ; label-based routing|

### Dataflow Architecture

Refer to [Heterogeneous Systems](Software.md/#heterogeneous-systems).

| Year | Venue | Authors | Title | Tags |
|-|-|-|------------------|--------------|
| 2024 | Micro | Carnegie Mellon University | The TYR Dataflow Architecture: Improving Locality by Taming Parallelism | local tag space; token synchronization instruction |
| 2024 | Micro | University of California, Riverside | Sparsepipe: Sparse Inter-operator Dataflow Architecture with Cross-Iteration Reuse | producer-consumer reuse; cross-iteration reuse; sparsepipe architecture|

#### Data Mapping

| Year | Venue | Authors | Title | Tags |
|-|-|-|------------------|--------------|
| 2023 | ISCA | Tsinghua University | MapZero: Mapping for Coarse-grained Reconfigurable Architectures with Reinforcement Learning and Monte-Carlo Tree Search | GAT embedding; reinforcement learning; Monte-Carlo tree | 
| 2023 | VLSI | Indian Institute of Technology Kharagpur | Application Mapping Onto Manycore Processor Architectures Using Active Search Framework | pe numbering method; recurrent neural network |

#### Task Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|------------------|--------------|
| 2020 | TCAD | Arizona State University | Runtime Task Scheduling Using Imitation Learning for Heterogeneous Many-Core Systems | imitation learning; two-level scheduling |
| 2023 | ICCAD | Peking University | Memory-aware Scheduling for Complex Wired Networks with Iterative Graph Optimization | topology-aware pruning; integer linear programming |

### Reconfigurable Architecture

## Electronic Design Automation

### Design Space Exploration

| Year | Venue | Authors | Title | Tags |
|-|-|-|------------------|--------------|
| 2021 | HPCA | Georgia Institute of Technology | MAGMA: An Optimization Framework for Mapping Multiple DNNs on Multiple Accelerator Cores | genetic algorithm; fine-grained job partition |
| 2023 | ISCA | Tsinghua University | Inter-layer Scheduling Space Definition and Exploration for Tiled Accelerators | inter-layer encoding; temperal cut; spatial cut; RA tree |
| 2024 | HPCA | Tsinghua University | Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators | layer-centric encoding; SA algorithm |

## Performance Evaluation

### Modeling and Simulation

| Year | Venue | Authors | Title | Tags |
|-|-|-|------------------|--------------|
| 2020 | Micro | Georgia Tech; NVIDIA | MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings |  data-centric; TemperalMap; SpatialMap; Cluster | 
| 2022 | OSDI | UC Berkeley | Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning | inter-operator parallelisms; intra-operator parallelisms | ASTRA-sim2.0: Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale | multi-dimensional heterogeneous topology; graph-based training-loop |
| 2023 | Micro | Peking University | TileFlow: A Framework for Modeling Fusion Dataflow via Tree-based Analysis | tree-based description; tile-centric |
| 2023 | ISPASS | Georgia Institute of Technology | ASTRA-sim2.0: Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale | graph-based training-loop; multi-dimensional heterogeneous topology |
| 2024 | ISCA | Stanford University | The Dataflow Abstract Machine Simulator Framework | communicating sequential processes; event-queue free; time multiplexing |

### Performance Analysis

| Year | Venue | Authors | Title | Tags |
|-|-|-|------------------|--------------|
| 2022 | ASPLOS | NC State University | ValueExpert: Exploring Value Patterns in GPU-accelerated Applications value-related inefficiencies | GPU-CPU data movement; redundant instructions | 