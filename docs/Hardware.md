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

#### NDP: DIMM

##### Benchmark

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | ATC | UBC | A Case Study of Processing-in-Memory in off-the-Shelf Systems | benchmark |
| 2022 | IEEE Access | ETH | Benchmarking a New Paradigm: Experimental Analysis and Characterization of a Real Processing-in-Memory System | benchmark suite "PrIM" |
| 2024 | CAL | KAIST | Analysis of Data Transfer Bottlenecks in Commercial PIM Systems: A Study With UPMEM-PIM | low MLP; manual data placement; unbalanced thread allocation and scheduling |
| 2024 | HPCA | KAIST | Pathfinding Future PIM Architectures by Demystifying a Commercial PIM Technology | simulator "uPIMulator" |

#### NDP: 3D-stacked DRAM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | DAC | SNU | MoNDE: Mixture of Near-Data Experts for Large-Scale Sparse Models | NDP for MoE; activation movement; GPU-MoNDE load-balancing scheme |
| 2024 | ASPLOS | PKU | SpecPIM: Accelerating Speculative Inference on PIM-Enabled System via Architecture-Dataflow Co-Exploration | algorithmic and architectural heterogeneity; PIM resource allocation; multi-model collaboration workflow |

#### CIM: SRAM

#### CIM: RRAM

##### RRAM CiM Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | ASPLOS | Purdue & HP | PUMA: A Programmable Ultra-efficient Memristor-based Accelerator for Machine Learning Inference | Programmable and general-purpose ReRAM based ML Accelerator; Supports an instruction set; Has protential for DNN training; Provides simulator that accepts model |
| 2018 | ICRC | Purdue & HP | Hardware-Software Co-Design for an Analog-Digital Accelerator for Machine Learning | compiler to translate model to ISA; ONNX interpreter to support models in common DL frame work; simulator to evaluate performance |
| 2023 | GLSVLSI | Yale | Examining the Role and Limits of Batchnorm Optimization to Mitigate Diverse Hardware-noise in In-memory Computing | non-idealities; circuit-level parasitic resistances and device-level non-idealities; crossbar-aware fine-tuning of batchnorm parameters |
| 2019 | ASPDAC | POSTECH | In-memory batch-normalization for resistive memory based binary neural network hardware | in-memory batchnormalization schemes; integrate BN layers on crossbar |

##### RRAM CiM: Convolutional layer

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | Nature | THU | Fully hardware-implemented memristor convolutional neural network | fabrication of high-yield, high-performance and uniform memristor crossbar arrays; hybrid-
training method; replication of multiple identical kernels for processing different inputs in parallel |
| 2020 | TCAS-I | Georgia Tech | Optimizing Weight Mapping and Data Flow for Convolutional Neural Networks on Processing-in-Memory Architectures | weight mapping to avoid multiple access to input; pipeline architecture for conv layer calculation |
| 2019 | TED | PKU | Convolutional Neural Networks Based on RRAM Devices for Image Recognition and Online Learning Tasks | RRAM-based hardware implementation of CNN; expand kernel to the size of image |
| 2021 | TCAD | SJTU | Efficient and Robust RRAM-Based Convolutional Weight Mapping With Shifted and Duplicated Kernel | shift and duplicate kernel (SDK) convolutional weight mapping architecture; parallel-window size allocation algorithm; kernel synchronization method |



##### RRAM Non-ideal Effects

Non-ideal Aware Methods: data types, training algiruthm, SRAM for compensation. Refer to [Data Type](#data-type).

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | DATE | Georgia Tech | Design of Reliable DNN Accelerator with Un-reliable ReRAM | dynamical fixed point data representation format; device variation aware training methodology |
| 2020 | DAC | ASU | Accurate Inference with Inaccurate RRAM Devices: Statistical Data, Model Transfer, and On-line Adaptation | introduce statistical variations in knowledge distillation; On-line sparse adaptation with a small SRAM array |
| 2020 | DATE | SJTU | Go Unary: A Novel Synapse Coding and Mapping Scheme for Reliable ReRAM-based Neuromorphic Computing | unary coding; priority mapping* |
| 2022 | TCAD | ASU | Hybrid RRAM/SRAM in-Memory Computing for Robust DNN Acceleration | integrates an RRAM-based IMC macro with a digital SRAM macro using a programmable shifter to compensate for RRAM variations; ensemble learning |

#### CIM: Hybrid Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Science | NTHU | Fusion of memristor and digital compute-in-memory processing for energy-efficient edge computing | Fusion of ReRAM and SRAM CiM; ReRAM SLC & MLC Hybrid; Current quantization; Weight shifting with compensation |

#### NVM

## Computer Architecture

### Data Type

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2017 | SFI | NUS | Beating Floating Point at its Own Game: Posit Arithmetic | Data type for universial number; Replacement for float; Highly adjustable; Dynamic range; regime, exponent and mantissa bits |
| 2021 | TCAS-II | Ashoka University | Fixed-Posit: A Floating-Point Representation for Error-Resilient Applications | the number of regime and exponent bits are fixed; a design of fixedposit multiplier; |
| 2022 | MICRO | SJTU | ANT: Exploiting Adaptive Numerical Data Type for Low-bit Deep Neural Network Quantization | fixed-length adaptive numerical data type; combines the advantages of float and int for adapting to the importance of different values within a tensor; adaptive framework that selects the best type for each tensor |
| 2024 | TCAD | HKU | DyBit: Dynamic Bit-Precision Numbers for Efficient Quantized Neural Network Inference | adaptive data representation with variablelength encoding; hardware-aware quantization framework |


### Domain-specific Accelerators

#### LLM Inference Accelerators

#### Graph Accelerators

### Memory Architecture

Refer to [Storage Systems](Software.md/#storage-systems).

#### Cache

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2006 | MICRO | Intel | Molecular Caches: A caching structure for dynamic creation of application-specific Heterogeneous cache regions | molecular cache architecture; application space identifier based cache partition; randy replacement algorithm |
| 2011 | ISCA | Stanford University | Vantage: Scalable and Efficient Fine-Grain Cache Partitioning | managed-unmanaged region division; churn-based management; feedback-based aperture control |

#### Heterogeneous Architecture
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2012 | HPCA | Georgia Institute of Technology | TAP: A TLP-Aware Cache Management Policy for a CPU-GPU Heterogeneous Architecture | thread-level parallelism; core sampling for cache effort indentification; cache block lifetime normalization; TAP-UCP for CPU; TAP-RRIP for GPU |
| 2017 | TACO | Intel | HAShCache: Heterogeneity-Aware Shared DRAMCache for Integrated Heterogeneous Systems | heterogeneity-aware DRAMCache scheduling PrIS; temporal bypass ByE; spatial occupancy control chaining |
| 2018 | ICS | NC State  | ProfDP: A Lightweight Profiler to Guide Data Placement in Heterogeneous Memory Systems | latency sensitivity; bandwidth sensitivity; moving factor based data placement |
| 2023 | HPCA | Tsinghua University | Baryon: Efficient Hybrid Memory Management with Compression and Sub-Blocking | stage area and selective commit for stable block; dual-format metadata scheme; cacheline-aligned compression and two-level replacements |
| 2024 | SC | Tsinghua University | Hydrogen: Contention-Aware Hybrid Memory for Heterogeneous CPU-GPU Architectures | fast memory decoupled partitioning; token-based slow memory migration; epoch-based sampling method; consistent hashing based reconfiguration |

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
| 2023 | MICRO | Tsinghua University | Heterogeneous Die-to-Die Interfaces: Enabling More Flexible Chiplet Interconnection Systems | heterogeneous interface hetero-PHY and hetero-channel; hetero-channel routing algorithm; application-aware scheduling |
| 2024 | MICRO | Tsinghua University | Ring Road: A Scalable Polar-Coordinate-based 2D Network-on-Chip Architecture | Ring Road topology based on isolated cycles and trees; polar coordinate DOR(dimension-order-routing); inter/intra-chip decouple routing algorithm |

#### Router

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2022 | HPCA | Chalmers | FastTrackNoC: A NoC with FastTrack Router Datapaths | non-turning hops; direct FastTrack flit path; zero-load latency analysis |
| 2023 | HPCA | Tsinghua University | A Scalable Methodology for Designing Efficient Interconnection Network of Chiplets | interface grouping; hypercube construction algorithm; deadlock-free adaptive routing algorithm; safe/unsafe flow control; network interleaving method|

### Dataflow Architecture

Refer to [Heterogeneous Systems](Software.md/#heterogeneous-systems).

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | ASPLOS | Tsinghua University | Tangram: Optimized Coarse-Grained Dataflow for Scalable NN Accelerators | buffer sharing dataflow(BSD); alternate layer loop ordering (ALLO) dataflow; heuristics spatial layer mapping algorithm |
| 2024 | MICRO | Carnegie Mellon University | The TYR Dataflow Architecture: Improving Locality by Taming Parallelism | local tag spaces technique; space tag managing instruction set; CT based concurrent-block communication |
| 2024 | MICRO | UC Riverside | Sparsepipe: Sparse Inter-operator Dataflow Architecture with Cross-Iteration Reuse | producer-consumer reuse; cross-iteration reuse; sub-tensor dependency; OEI dataflow; sparsepipe architecture|

#### Data Mapping

##### Heuristic Algorithm

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | HPCA | Georgia Tech | MAGMA: An Optimization Framework for Mapping Multiple DNNs on Multiple Accelerator Cores | sub-accelerator selection; fine-grained job prioritization; MANGA crossover genetic operators |
| 2023 | ISCA | Tsinghua University | MapZero: Mapping for Coarse-grained Reconfigurable Architectures with Reinforcement Learning and Monte-Carlo Tree Search | GAT based DFG and CGRA embedding; routing penalty based reinforcement learning; Monte-Carlo tree search space exploration |
| 2023 | VLSI | IIT Kharagpur | Application Mapping Onto Manycore Processor Architectures Using Active Search Framework | RNN based active search framework; IP-Core Numbering Scheme; active search with/without pretraining |
| 2024 | HPCA | Tsinghua University | Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators | layer-centric encoding method; DP-based graph partition algorithm; SA based D2D link communication optimization |

##### Optimization Modeling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | FPGA | ETH Zurich | Flexible Communication Avoiding Matrix Multiplication on FPGA with High-Level Synthesis | computation and I/O decomposition model for matrix multiplication; 1D array collapse mapping method; internal double buffering |
| 2021 | HPCA | Georgia Tech | Heterogeneous Dataflow Accelerators for Multi-DNN Workloads | heterogeneous dataflow accelerators (HDAs) for DNN; dataflow flexibility; high utilization across the sub-accelerators |
| 2023 | MICRO | Alibaba; CUHK | ArchExplorer: Microarchitecture Exploration Via Bottleneck Analysis | dynamic event-dependence graph(EDG); induced DEG based critical path construction; bottleneck-removal-driven DSE |
| 2023 | ISCA | Tsinghua University | Inter-layer Scheduling Space Definition and Exploration for Tiled Accelerators | inter-layer encoding method; temperal cut; spatial cut; RA tree analysis |


#### Task Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | TCAD | Arizona State University | Runtime Task Scheduling Using Imitation Learning for Heterogeneous Many-Core Systems | offline Oracle optimizaion strategy; hierarchical imitation learning based scheduling; two-level scheduling |
| 2023 | ICCAD | Peking University | Memory-aware Scheduling for Complex Wired Networks with Iterative Graph Optimization | topology-aware pruning algorithm; integer linear programming scheduling method; sub-graph fusion algorithm ; memory-aware graph partitioning|
| 2023 | MICRO | Duke University | Si-Kintsugi: Towards Recovering Golden-Like Performance of Defective Many-Core Spatial Architectures for AI | graph alignment algoithm for dataflow graph and platform pe grap; producer-consumer pattern dataflow generation algorithm |

### Reconfigurable Architecture

### many-core architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | MICRO | Tsinghua University | MAICC : A Lightweight Many-core Architecture with In-Cache Computing for Multi-DNN Parallel Inference | slice improved and hardware-implemented reduction CIM; ISA extension for CIM; CNN layer segmentation and mapping algorithm |
| 2023 | MICRO | Yonsei University | McCore: A Holistic Management of High-Performance Heterogeneous Multicores | cluster partitioning via index hash function; partitions balancing method; hardware support for RL based scheduling |

## Electronic Design Automation

## Performance Evaluation

### Modeling and Simulation

#### Dataflow Architectrue

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|

| 2022 | OSDI | UC Berkeley | Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning | inter-operator parallelisms; intra-operator parallelisms; ILP and DP hierarchical optimization |
| 2023 | MICRO | Peking University | TileFlow: A Framework for Modeling Fusion Dataflow via Tree-based Analysis | 3D design space of fusion dataflow; tree-based description; tile-centric notation |
| 2024 | ISCA | Stanford University | The Dataflow Abstract Machine Simulator Framework | communicating sequential processes; event-queue free execution; context-channel based description; asynchronous distributed time |

#### Connection Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | MICRO | Georgia Institute of Technology; NVIDIA | MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings | data-centric mapping; data reuse analysis; TemperalMap; SpatialMap; analytical cost model |
| 2023 | ISPASS | Georgia Institute of Technology | ASTRA-sim2.0: Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale | graph-based training-loop execution; multi-dimensional heterogeneous topology construction; analytical network backend |
| 2024 | ATC | Tsinghua University | Evaluating Chiplet-based Large-Scale Interconnection Networks via Cycle-Accurate Packet-Parallel Simulation | packet-centric simulation; critical resources recorading for process-order-induced deviations; unimportant stages elimination |

### Performance Analysis

#### Redundancy Detection

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | SC | NC State | ZeroSpy: Exploring Software Inefficiency with Redundant Zeros | code-centric analysis for instruction detection; data-centric analysis for data detection |
| 2020 | SC | NC State | GVPROF: A Value Profiler for GPU-Based Clusters | temporal/spatial load/store redundancy; hierarchical sampling for reducing monitoring overhead; bidirectional search algorithm on dependency graph |
| 2022 | ASPLOS | NC State | ValueExpert: Exploring Value Patterns in GPU-accelerated Applications value-related inefficiencies | data value pattern recoginition; value flow graph; parallel intervals merging algorithm |
| 2022 | SC | NC State | Graph Neural Networks Based Memory Inefficiency Detection Using Selective Sampling | dead store; silent store; silent load; assembly-level procedural control-flow embedding; dynamic value semantic embedding; relative positional encoding for different compilation options |

#### Stall Attribution

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | ICPE | NC State University | DrGPU: A Top-Down Profiler for GPU | device memory stall; synchronization stall; instruction related stall; shared memory related stall |
| 2024 | MICRO | NUDT | HyFiSS: A Hybrid Fidelity Stall-Aware Simulator for GPGPUs | memory/compute structual/data stall; synchronization stall; control stall; idle stall; cooperative thread array-sets based SM sampling algorithm |
