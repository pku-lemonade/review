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

##### Communication
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | HPCA | PKU | DIMM-Link: Enabling Efficient Inter-DIMM Communication for Near-Memory Processing | high-speed hardware link bridges between DIMMs; direct intra-group P2P communication & broadcast; hybrid routing mechanism for inter-group communication |
| 2024 | ISCA | THU | NDPBridge: Enabling Cross-Bank Coordination in Near-DRAM-Bank Processing Architectures | gather & scatter messages via buffer chip; task-based message-passing model; hierarchical, data-transfer-aware load balancing |

##### Benchmark

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | ATC | UBC | A Case Study of Processing-in-Memory in off-the-Shelf Systems | benchmark |
| 2022 | IEEE Access | ETH | Benchmarking a New Paradigm: Experimental Analysis and Characterization of a Real Processing-in-Memory System | benchmark suite "PrIM" |
| 2024 | CAL | KAIST | Analysis of Data Transfer Bottlenecks in Commercial PIM Systems: A Study With UPMEM-PIM | low MLP; manual data placement; unbalanced thread allocation and scheduling |
| 2024 | IEEE Access | Univ. of Lisbon | NDPmulator: Enabling Full-System Simulation for Near-Data Accelerators From Caches to DRAM | simulator "PiMulator" based on Ramulator & gem5; full system support; multiple ISA support |
| 2024 | HPCA | KAIST | Pathfinding Future PIM Architectures by Demystifying a Commercial PIM Technology | simulator "uPIMulator" |

#### NDP: CXL

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2022 | MICRO | UCSB | BEACON: Scalable Near-Data-Processing Accelerators for Genome Analysis near Memory Pool with the CXL Support | scalable hardware accelerator inside CXL switch or bank | lossless memory expansion for CXL memory pools |
| 2024 | ICS | Samsung | CLAY: CXL-based Scalable NDP Architecture Accelerating Embedding Layers | direct interconnect between DRAM clusters; dedicated memory address mapping scheme; Multi-CLAY system support through customized CXL switch |
| 2024 | MICRO | SK Hyrix | Low-overhead General-purpose Near-Data Processing in CXL Memory Expanders | CXL.mem protocol instead of CXL.io (DMA) for low-latency; "lightweight" threads to reduce address calculation overhead |

#### NDP: 3D-stacked DRAM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2013 | PACT | KAIST | Memory-centric System Interconnect Design with Hybrid Memory Cubes | memory-centric network; distributor-based topology for reduced latency; non-minimal routing for higher throughput | 
| 2024 | DAC | SNU | MoNDE: Mixture of Near-Data Experts for Large-Scale Sparse Models | NDP for MoE; activation movement; GPU-MoNDE load-balancing scheme |
| 2024 | ASPLOS | PKU | SpecPIM: Accelerating Speculative Inference on PIM-Enabled System via Architecture-Dataflow Co-Exploration | algorithmic and architectural heterogeneity; PIM resource allocation; multi-model collaboration workflow |

##### Benchmark

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | DAC | ETHZ | NAPEL: Near-Memory Computing Application Performance Prediction via Ensemble Learning | simulator "Ramulator-PIM"; tracefile from Ramulator & run on zsim |
| 2021 | CAL | Univ. of Virginia | MultiPIM: A Detailed and Configurable Multi-Stack Processing-In-Memory Simulator | simulator "MultiPIM"; multi-stack & virtual memory support; parallel offloading |

#### General CiM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | ISVLSI | USC | Multi-Objective Neural Architecture Search for In-Memory Computing | neural architecture search methodology; integration of Hyperopt, PyTorch and MNSIM |
| 2024 | ISPASS | MIT | CiMLoop: A Flexible, Accurate, and Fast Compute-In-Memory Modeling Tool | flexible specification to describe CiM systems; accurate model/fast statistical model of data-value-dependent component energy |
| 2018 | TCAD | ASU | NeuroSim: A Circuit-Level Macro Model for Benchmarking Neuro-Inspired Architectures in Online Learning | estimate the circuit-level performance of neuro-inspired architectures; estimates the area, latency, dynamic energy, and leakage power; Support both SRAM and eNVM; tested on 2-layer MLP NN, MNIST |
| 2019 | IEDM | Geogria Tech | DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies | a python wrapper to interface NeuroSim; for inference only |
| 2020 | TCAD | ZJU | Eva-CiM: A System-Level Performance and Energy Evaluation Framework for Computing-in-Memory Architectures | models for capturing memory access and dependency-aware ISA traces; models for quantifying interactions between the host CPU and the CiM module |


#### CIM: SRAM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | TCASAI | Purdue | Algorithm Hardware Co-Design for ADC-Less Compute In-Memory Accelerator | reduce ADC overhead in analog CiM architectures; Quantization-Aware Training; Partial Sum Quantization; ADC-Less hybrid analog-digital CiM hardware architecture HCiM |
| 2024 | ISCAS | NYCU | CIMR-V: An End-to-End SRAM-based CIM  Accelerator with RISC-V for AI Edge Device | incorporates CIM layer fusion, convolution/max pooling pipeline, and weight fusion; weight fusion: pipelining the CIM convolution and weight loading |
| 2021 | TCAD | Geogria Tech | DNN+NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for On-Chip Training | non-ideal device properties of NVMS' effect for on-chip training |
| 2020 | ISCAS | JCU | MemTorch: A Simulation Framework for Deep Memristive Cross-Bar Architectures | supports both GPUs and CPUs; integrates directly with PyTorch; simulate non-idealities of memristive devices within cross-bar, tested on VGG-16, CIFAR-10 |


#### CIM: RRAM


##### RRAM CiM: Architecture


| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | ASPLOS | Purdue & HP | PUMA: A Programmable Ultra-efficient Memristor-based Accelerator for Machine Learning Inference | Programmable and general-purpose ReRAM based ML Accelerator; Supports an instruction set; Has protential for DNN training; Provides simulator that accepts model |
| 2018 | ICRC | Purdue & HP | Hardware-Software Co-Design for an Analog-Digital Accelerator for Machine Learning | compiler to translate model to ISA; ONNX interpreter to support models in common DL frame work; simulator to evaluate performance |
| 2024 | DATE | UCAS | PIMSIM-NN: An ISA-based Simulation Framework for Processing-in-Memory Accelerators | event-driven simulation approach; can evaluate the optimizations of software and hardware independently |
| 2018 | TCAD | THU | MNSIM: Simulation Platform for Memristor-Based Neuromorphic Computing System | reference design for largescale neuromorphic accelerator and can also be customized; behavior-level computing accuracy model |
| 2023 | TCAD | THU | MNSIM 2.0: A Behavior-Level Modeling Tool for Processing-In-Memory Architectures | integrated PIM-oriented NN model training and quantization flow; unified PIM memory array model; support for mixed-precision NN operations |

##### RRAM CiM: Architecture optimization

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | MICRO | HUST | DRCTL: A Disorder-Resistant Computation  Translation Layer Enhancing the Lifetime and  Performance of Memristive CIM Architecture | address conversion method for dynamic scheduling; hierarchical wear-leveling (HWL) strategy for reliability improvement; data layout-aware selective remapping (LASR) to improve communication locality and reduce latency |
| 2024 | DATE | RWTH Aachen University | CLSA-CIM: A Cross-Layer Scheduling Approach for Computing-in-Memory Architectures | algorithm to decide which parts of NN are duplicated to reduce inference latency; crosslayer scheduling on tiled CIM architectures |
| 2024 | TC | SJTU | ERA-BS: Boosting the Efficiency of ReRAM-Based  PIM Accelerator With Fine-Grained  Bit-Level Sparsity | bitlevel sparsity in both weights and activations; bit-flip scheme; dynamic activation sparsity exploitation scheme |



##### RRAM CiM: BatchNorm layer

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
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
| 2024 | LATS | AMU | Analysis of Conductance Variability in RRAM for  Accurate Neuromorphic Computing | analyzation and quantification of conductance variability in RRAMs; analysis of conductance variation over multiple cycles |

#### CIM: Hybrid Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Science | NTHU | Fusion of memristor and digital compute-in-memory processing for energy-efficient edge computing | Fusion of ReRAM and SRAM CiM; ReRAM SLC & MLC Hybrid; Current quantization; Weight shifting with compensation |
| 2024 | IPDPS | Georgia Tech | Harmonica: Hybrid Accelerator to Overcome Imperfections of Mixed-signal DNN Accelerators | select and transfer imperfectionsensitive weights to digital accelerator; hybrid quantization(weights on analog part is more quantized) |

#### CIM: Quantization

| 2024 | TCAD | BUAA | CIMQ: A Hardware-Efficient Quantization Framework for Computing-In-Memory-Based Neural Network Accelerators | bit-level sparsity induced activation quantization; quantizing partial sums to decrease required resolution of ADCs; arraywise quantization granularity |
| 2018 | CVPR | Google | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | integer-only inference arithmetic; quantizesh both weights and activations as 8-bit integers, bias 32-bit; provides both quantized inference framework and training frame work |
| 2024 | TCAD | BUAA | CIM²PQ: An Arraywise and Hardware-Friendly Mixed Precision Quantization Method for Analog Computing-In-Memory | mixed precision quantization method based on evolutionary algorithm; arraywise quantization granularity; evaluation method to obtain the performance of strategy on the CIM |

#### NVM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | ISCAS | UMCP | On-Chip Adaptation for Reducing Mismatch in Analog Non-Volatile Device Based Neural Networks | float-gate transistors based; hot-electron injection to address the issue of mismatch and variation |



## Computer Architecture

### Data Type

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2017 | SFI | NUS | Beating Floating Point at its Own Game: Posit Arithmetic | Data type for universial number; Replacement for float; Highly adjustable; Dynamic range; regime, exponent and mantissa bits |
| 2021 | TCAS-II | Ashoka University | Fixed-Posit: A Floating-Point Representation for Error-Resilient Applications | the number of regime and exponent bits are fixed; a design of fixedposit multiplier; |
| 2022 | MICRO | SJTU | ANT: Exploiting Adaptive Numerical Data Type for Low-bit Deep Neural Network Quantization | fixed-length adaptive numerical data type; combines the advantages of float and int for adapting to the importance of different values within a tensor; adaptive framework that selects the best type for each tensor |
| 2024 | TCAD | HKU | DyBit: Dynamic Bit-Precision Numbers for Efficient Quantized Neural Network Inference | adaptive data representation with variablelength encoding; hardware-aware quantization framework |
| 2024 | Arxiv | Harvard | Nanoscaling Floating-Point (NxFP): NanoMantissa, Adaptive Microexponents, and Code Recycling for Direct-Cast Compression of Large Language Models | Nanoscaling Floating-Point (NxFP); NanoMantissa; Adaptive Microexponents; Code Recycling |


### Domain-specific Accelerators

#### DNN Accelerators

##### Layer Fusion Accelerators

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2016 | MICRO | SBU | Fused-Layer CNN Accelerators | fuse the processing of multiple CNN layers by modifying the order in which the input data are brought on chip |
| 2025 | TC | KU Leuven | Stream: Design Space Exploration of Layer-Fused DNNs on Heterogeneous Dataflow Accelerators | fine-grain mapping paradigm; mapping of layer-fused DNNs on heterogeneous dataflow accelerator architectures; memory- and communication-aware latency analysis; constraint optimization |
| 2024 | SOCC | IIT Hyderabad | Hardware-Aware Network Adaptation using Width and Depth Shrinking including Convolutional and Fully Connected Layer Merging | “Width Shrinking”: reduces the number of feature maps in CNN layers; “Depth Shrinking”: Merge of conv layer and fc layer |
| 2024 | ICSAI | MIT | LoopTree: Exploring the Fused-Layer Dataflow  Accelerator Design Space | design space that supports set of tiling, recomputation, retention choices, and their combinations; model that validates design space |

##### LLM Inference Accelerators

#### Graph Accelerators

### Memory Architecture

Refer to [Storage Systems](Software.md/#storage-systems).

#### Cache

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2006 | MICRO | Intel | Molecular Caches: A caching structure for dynamic creation of application-specific Heterogeneous cache regions | molecular cache architecture; application space identifier based cache partition; randy replacement algorithm |
| 2011 | ISCA | Stanford University | Vantage: Scalable and Efficient Fine-Grain Cache Partitioning | managed-unmanaged region division; churn-based management; feedback-based aperture control |
| 2016 | HPCA | Intel | Cache QoS: From concept to reality in the Intel® Xeon® processor E5-2600 v3 product family | cache monitoring technology; cache allocation technology; resource monitoring IDs (RMIDs); classes of service (CLOS) |
| 2018 | EuroSys | PKU | DCAPS: dynamic cache allocation with partial sharing | dynamic fine-grained shared cache management; balance cache utilization and contention; online practical miss rate curve |

#### Heterogeneous Architecture
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2012 | HPCA | Georgia Institute of Technology | TAP: A TLP-Aware Cache Management Policy for a CPU-GPU Heterogeneous Architecture | thread-level parallelism; core sampling for cache effort indentification; cache block lifetime normalization; TAP-UCP for CPU; TAP-RRIP for GPU |
| 2017 | TACO | Intel | HAShCache: Heterogeneity-Aware Shared DRAMCache for Integrated Heterogeneous Systems | heterogeneity-aware DRAMCache scheduling PrIS; temporal bypass ByE; spatial occupancy control chaining |
| 2018 | ICS | NC State  | ProfDP: A Lightweight Profiler to Guide Data Placement in Heterogeneous Memory Systems | latency sensitivity; bandwidth sensitivity; moving factor based data placement |
| 2023 | HPCA | Tsinghua University | Baryon: Efficient Hybrid Memory Management with Compression and Sub-Blocking | stage area and selective commit for stable block; dual-format metadata scheme; cacheline-aligned compression and two-level replacements |
| 2024 | SC | Tsinghua University | Hydrogen: Contention-Aware Hybrid Memory for Heterogeneous CPU-GPU Architectures | fast memory decoupled partitioning; token-based slow memory migration; epoch-based sampling method; consistent hashing based reconfiguration |

#### Prefetcher

##### LLM Inference Prefetching

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | Arxiv | Huawei Zurich | PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving | computational graph-based prefetching; prefetch KV cache to L2 cache; optimal hardware design for prefetching |

#### DRAM

### Communication Architecture

Refer to [Distributed Systems](Software.md/#distributed-systems).

#### Network-on-Chip

##### Wafer-Scale

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | SC | Tsinghua University | Switch-Less Dragonfly on Wafers: A Scalable Interconnection Architecture based on Wafer-Scale Integration | four-level topology structure; minimal routing algorithm on dragonfly for VC vumber reduction | 

##### Topology

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | HPCA | George Washington University | Adapt-NoC: A Flexible Network-on-Chip Design for Heterogeneous Manycore Architectures | mux based adaptable router architecture; adaptable link design; reinforcement learning based subNoC optimization algorithm |
| 2022 | HPCA | Huawei | Application Defined On-chip Networks for Heterogeneous Chiplets: An Implementation Perspective | bufferless multi-ring NoC design; application-architecture-physical co-design method; architecture expressiveness; deadlock resolution SWAP mechanism |
| 2024 | MICRO | Tsinghua University | Ring Road: A Scalable Polar-Coordinate-based 2D Network-on-Chip Architecture | Ring Road topology based on isolated cycles and trees; polar coordinate DOR(dimension-order-routing); inter/intra-chip decouple routing algorithm |
| 2024 | Arxiv | Washington State University | Atleus: Accelerating Transformers on the Edge Enabled by 3D Heterogeneous Manycore Architectures | heterogeneous 3D 
NoC; pipeline design across heterogeneous resources; crossbar-wise quantization |

##### Interconnect

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2012 | SIGCOMM | Carnegie Mellon University | On-Chip Networks from a Networking Perspective: Congestion and Scalability in Many-Core Interconnects | congestion control mechanism for bufferless NoC; interval-based congestion control algorithm; simple injection throttling algorithm |
| 2023 | ICCAD | University of Central Florida | ARIES: Accelerating Distributed Training in Chiplet-based Systems via Flexible Interconnects | directional bypassing link; ARIES link with transistor; ARIES all-reduce optimization algorithm |
| 2023 | MICRO | Tsinghua University | Heterogeneous Die-to-Die Interfaces: Enabling More Flexible Chiplet Interconnection Systems | heterogeneous interface hetero-PHY and hetero-channel; hetero-channel routing algorithm; application-aware scheduling |

##### Processing on NoC

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | HPCA | Drexel University | SnackNoC: Processing in the Communication Layer | communication fabric quantification; central packet manager for instruction flit; router compute unit as dataflow pe |

#### Router

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2016 | HPCA | KTH Royal Institute of Technology | DVFS for NoCs in CMPs: A Thread Voting Approach | thread voting based DVFS machenism; pre-defined region-based V/F adjustment algorithm |
| 2022 | HPCA | Chalmers | FastTrackNoC: A NoC with FastTrack Router Datapaths | non-turning hops; direct FastTrack flit path; zero-load latency analysis |
| 2022 | HPCA | University of Toronto | Stay in your Lane: A NoC with Low-overhead Multi-packet Bypassing | FastFlow flow controll method; time-division-multiplexed (TDM) based non-overlapping FastPass-lanes; FastPass for throughput enhancement |
| 2023 | HPCA | Tsinghua University | A Scalable Methodology for Designing Efficient Interconnection Network of Chiplets | interface grouping; hypercube construction algorithm; deadlock-free adaptive routing algorithm; safe/unsafe flow control; network interleaving method|

### Dataflow Architecture

Refer to [Heterogeneous Systems](Software.md/#heterogeneous-systems).

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | ASPLOS | Tsinghua University | Tangram: Optimized Coarse-Grained Dataflow for Scalable NN Accelerators | buffer sharing dataflow(BSD); alternate layer loop ordering (ALLO) dataflow; heuristics spatial layer mapping algorithm |
| 2024 | MICRO | Carnegie Mellon University | The TYR Dataflow Architecture: Improving Locality by Taming Parallelism | local tag spaces technique; space tag managing instruction set; CT based concurrent-block communication |
| 2024 | MICRO | UC Riverside | Sparsepipe: Sparse Inter-operator Dataflow Architecture with Cross-Iteration Reuse | producer-consumer reuse; cross-iteration reuse; sub-tensor dependency; OEI dataflow; sparsepipe architecture|

#### Data Mapping

##### Servey

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2013 | DAC | NUS | Mapping on Multi/Many-core Systems: Survey of Current and Emerging Trends | dense/run-time mapping; centralized/distributred management; hybrid mapping |

##### Heuristic Algorithm

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | HPCA | Georgia Tech | MAGMA: An Optimization Framework for Mapping Multiple DNNs on Multiple Accelerator Cores | sub-accelerator selection; fine-grained job prioritization; MANGA crossover genetic operators |
| 2023 | ISCA | Tsinghua University | MapZero: Mapping for Coarse-grained Reconfigurable Architectures with Reinforcement Learning and Monte-Carlo Tree Search | GAT based DFG and CGRA embedding; routing penalty based reinforcement learning; Monte-Carlo tree search space exploration |
| 2023 | VLSI | IIT Kharagpur | Application Mapping Onto Manycore Processor Architectures Using Active Search Framework | RNN based active search framework; IP-Core Numbering Scheme; active search with/without pretraining |
| 2024 | HPCA | Tsinghua University | Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators | layer-centric encoding method; DP-based graph partition algorithm; SA based D2D link communication optimization |
| 2024 | ASPLOS | Tsinghua University | Cocco: Hardware-Mapping Co-Exploration towards Memory Capacity-Communication Optimization | consumption-centric flow based subgraph execution scheme; main/side region based memory management |

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
| 2017 | HPCA | Ghent University | Reliability-Aware Scheduling on Heterogeneous Multicore Processors | core reliability characteristics difference; system soft error rate; sampling-based reliability-aware scheduling algorithm |
| 2020 | TCAD | Arizona State University | Runtime Task Scheduling Using Imitation Learning for Heterogeneous Many-Core Systems | offline Oracle optimizaion strategy; hierarchical imitation learning based scheduling; two-level scheduling |
| 2023 | ICCAD | Peking University | Memory-aware Scheduling for Complex Wired Networks with Iterative Graph Optimization | topology-aware pruning algorithm; integer linear programming scheduling method; sub-graph fusion algorithm ; memory-aware graph partitioning|
| 2023 | MICRO | Duke University | Si-Kintsugi: Towards Recovering Golden-Like Performance of Defective Many-Core Spatial Architectures for AI | graph alignment algoithm for dataflow graph and platform pe grap; producer-consumer pattern dataflow generation algorithm |

### Reconfigurable Architecture

### Many-core Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2015 | HPCA | Cornel University | Increasing Multicore System Efficiency through Intelligent Bandwidth Shifting | online bandwidth shifting mechanism; prefetch usefulness (PU) level |
| 2015 | HPCA | IBM | XChange: A Market-based Approach to Scalable Dynamic Multi-resource Allocation in Multicore Architectures | CMP multiresource allocation mechanism XChange; market framework based modeling |
| 2018 | MICRO | Seoul National University | RpStacks-MT: A High-throughput Design Evaluation Methodology for Multi-core Processors | graph-based multi-core performance model; distance-based memory system model; dynamic scheduling reconstruction method |
| 2023 | MICRO | Tsinghua University | MAICC : A Lightweight Many-core Architecture with In-Cache Computing for Multi-DNN Parallel Inference | slice improved and hardware-implemented reduction CIM; ISA extension for CIM; CNN layer segmentation and mapping algorithm |
| 2023 | MICRO | Yonsei University | McCore: A Holistic Management of High-Performance Heterogeneous Multicores | cluster partitioning via index hash function; partitions balancing method; hardware support for RL based scheduling |

#### Application Optimization

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | SC | NUDT | Optimizing Direct Convolutions on ARM Multi-Cores | direct convolution algorithm NDirect; loop ordering algorithm; micro convolution kernal for computing&packeting |
| 2023 | SC | NUDT | Optimizing MPI Collectives on Shared Memory Multi-Cores | intra-node reduction algorithm for redundant data movements; fine grained non-temporal store based adaptive collectives |
| 2024 | PPoPP | NUDT | Towards Scalable Unstructured Mesh Computations on Shared Memory Many-Cores | task dependency tree(TDT); tree traversal based parallel algorithm for CPU/GPU |


### Heterogeneous Many-core System

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2018 | ICCAD | Washington State University | Hybrid On-Chip Communication Architectures for Heterogeneous Manycore Systems | many-to-few communication patterns; long range shortcut based wireless NoC ; 3D-TSV based heterogeneous NoC |
| 2018 | IEEE TC | Washington State University | On-Chip Communication Network for Efficient Training of Deep Convolutional Networks on Heterogeneous Manycore Systems | wireless-enabled heterogeneous NoC; archived multi-objective simulated annealing for network connectivity |

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
| 2014 | JPDC | IN2P3 Computing Center | Versatile, scalable, and accurate simulation of distributed applications and platforms | API based communication&computation description; informed model of TCP for moderate size grids; file based modular network representation technique |
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

#### Error Pattern

##### Manycore Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2009 | MICRO | University of Illinois | mSWAT: Low-Cost Hardware Fault Detection and Diagnosis for Multicore Systems | selective Triple Modular Redundant(TMR) replay method; symptom based fault detection; permanent/transient fault |
| 2015 | IEEE TSM | NTU | Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets | wafer map failure pattern; wafer map similarity ranking; radon/geometry-based feature extraction; WM-811K wafer map dataset |

##### System Level

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2017 | SC | Argonne National Lab | Run-to-run Variability on Xeon Phi based Cray XC Systems | OS noise based core-level variability; tile-level varibility; memory mode varibility |
| 2018 | FAST | University of Chicago | Fail-Slow at Scale: Evidence of Hardware Performance Faults in Large Production Systems | conversion among fail-stop/slow/trasient; permanent/transient/partial slowdown; internal/external root causes |
| 2019 | ATC | University of Chicago | IASO: A Fail-Slow Detection and Mitigation Framework for Distributed Storage Services | slowdown detection based on peer score; sub-root causes for five kinds of root causes |

#### Hardware Fault

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2014 | DTIS | A Survey on Simulation-Based Fault Injection Tools for Complex Systems | runtime fault injection; compile-time fault injection |
| 2024 | Arxiv | George Washington University | Algorithmic Strategies for Sustainable Reuse of Neural Network Accelerators with Permanent Faults | stack-at-0/1 faults; weight register fault; invertible scaling and shifting technique; elementary tile operations for mantissa fault |