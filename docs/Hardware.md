# Hardware

## Emerging Technologies

### 3DIC

#### Thermal Evaluation
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | arXiv | SJTU | Cool-3D: An End-to-End Thermal-Aware Framework for Early-Phase Design Space Exploration of Microfluidic-Cooled 3DICs | end-to-end thermal-aware framework; microfluidic cooling integration; Pre-RTL design space exploration; floorplan designer; microfluidic cooling strategy generator |

#### Benchmarks

##### 3DIC Backend
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | arXiv | NJU | Open3DBench: Open-Source Benchmark for 3D-IC Backend Implementation and PPA Evaluation | open-source 3D-IC benchmark; modular 3D partitioning and placement; Open3D-DMP algorithm for cross-die co-placement; comprehensive PPA evaluation with thermal simulation |

### Chiplets

#### Survey

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | Electronics | NUDT | Chiplet Heterogeneous Integration Technology—Status and Challenges | heterogeneous integration technology; interconnect interfaces and protocols; packaging technology|
| 2022 | CCF THPC | ICT | Survey on chiplets: interface, interconnect and integration methodology | development history; interfaces and protocols; packaging technology; EDA tool; standardization of chiplet technology |
| 2024 | IEEE CASS | Tsinghua University | Chiplet Heterogeneous Integration Technology—Status and Challenges | wafer-scale chip architecture; compiler tool chain; integration technology; wafer-scale system; fault tolerance |

### Novel Memory Technologies

#### CXL
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | arXiv | CAS-ICT | CXL over Ethernet: A Novel FPGA-based Memory Disaggregation Design in Data Centers | "CXL over Ethernet architecture" for extending memory disaggregation; FPGA-based prototype with cache optimization; switch-independent congestion control algorithm; native memory semantics for transparent access; combining CXL and Ethernet for low-latency remote memory access |

#### NDP: DIMM

##### Communication
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | HPCA | PKU | DIMM-Link: Enabling Efficient Inter-DIMM Communication for Near-Memory Processing | high-speed hardware link bridges between DIMMs; direct intra-group P2P communication & broadcast; hybrid routing mechanism for inter-group communication |
| 2024 | ISCA | THU | NDPBridge: Enabling Cross-Bank Coordination in Near-DRAM-Bank Processing Architectures | gather & scatter messages via buffer chip; task-based message-passing model; hierarchical, data-transfer-aware load balancing |

#### PIM: (e)DRAM
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | ICCD | ASU | CIDAN: Computing in DRAM with Artificial Neurons | "Threshold Logic Processing Element (TLPE)" for in-memory computation; Four-bank activation window; Configurable threshold functions; Energy-efficient bitwise operations; Integration with DRAM architecture |
| 2025 | arXiv | KAIST | RED: Energy Optimization Framework for eDRAM-based PIM with Reconfigurable Voltage Swing and Retention-aware Scheduling | "RED framework" for energy optimization; reconfigurable eDRAM design; retention-aware scheduling; trade-off analysis between RBL voltage swing, sense amplifier power, and retention time; refresh skipping and sense amplifier power gating |


#### PIM & NDP: Benchmarks

##### Benchmarks for Conventional Computing

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | ATC | UBC | A Case Study of Processing-in-Memory in off-the-Shelf Systems | benchmark |
| 2022 | IEEE Access | ETH | Benchmarking a New Paradigm: Experimental Analysis and Characterization of a Real Processing-in-Memory System | benchmark suite "PrIM" |
| 2024 | CAL | KAIST | Analysis of Data Transfer Bottlenecks in Commercial PIM Systems: A Study With UPMEM-PIM | low MLP; manual data placement; unbalanced thread allocation and scheduling |
| 2024 | IEEE Access | Univ. of Lisbon | NDPmulator: Enabling Full-System Simulation for Near-Data Accelerators From Caches to DRAM | simulator "PiMulator" based on Ramulator & gem5; full system support; multiple ISA support |
| 2024 | HPCA | KAIST | Pathfinding Future PIM Architectures by Demystifying a Commercial PIM Technology | simulator "uPIMulator" |

##### Benchmarks for Quantum Computing
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | ASPDAC | NUS | PIMutation: Exploring the Potential of PIM Architecture for Quantum Circuit Simulation | "PIMutation framework" for quantum circuit simulation; gate merging optimization; row swapping instead of matrix multiplication; vector partitioning for separable states; leveraging UPMEM PIM architecture

#### NDP: CXL

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2022 | MICRO | UCSB | BEACON: Scalable Near-Data-Processing Accelerators for Genome Analysis near Memory Pool with the CXL Support | scalable hardware accelerator inside CXL switch or bank | lossless memory expansion for CXL memory pools |
| 2024 | ICS | Samsung | CLAY: CXL-based Scalable NDP Architecture Accelerating Embedding Layers | direct interconnect between DRAM clusters; dedicated memory address mapping scheme; Multi-CLAY system support through customized CXL switch |
| 2024 | MICRO | SK Hyrix | Low-overhead General-purpose Near-Data Processing in CXL Memory Expanders | CXL.mem protocol instead of CXL.io (DMA) for low-latency; "lightweight" threads to reduce address calculation overhead |
##### CXL-based Memory Pools
| 2023 | ASPLOS | Virginia Tech | Pond: CXL-Based Memory Pooling Systems for Cloud Platforms | "CXL-based memory pooling"; small-pool design for low latency; machine learning model for memory allocation prediction; zero-core virtual NUMA (zNUMA) node for untouched memory |

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

#### General CiM & PiM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | ISVLSI | USC | Multi-Objective Neural Architecture Search for In-Memory Computing | neural architecture search methodology; integration of Hyperopt, PyTorch and MNSIM |
| 2024 | ISPASS | MIT | CiMLoop: A Flexible, Accurate, and Fast Compute-In-Memory Modeling Tool | flexible specification to describe CiM systems; accurate model/fast statistical model of data-value-dependent component energy |
| 2018 | TCAD | ASU | NeuroSim: A Circuit-Level Macro Model for Benchmarking Neuro-Inspired Architectures in Online Learning | estimate the circuit-level performance of neuro-inspired architectures; estimates the area, latency, dynamic energy, and leakage power; Support both SRAM and eNVM; tested on 2-layer MLP NN, MNIST |
| 2019 | IEDM | Geogria Tech | DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies | a python wrapper to interface NeuroSim; for inference only |
| 2020 | TCAD | ZJU | Eva-CiM: A System-Level Performance and Energy Evaluation Framework for Computing-in-Memory Architectures | models for capturing memory access and dependency-aware ISA traces; models for quantifying interactions between the host CPU and the CiM module |
| 2025 | AICAS | Univ. of Virginia | Optimizing and Exploring System Performance in Compact Processing-in-Memory-based Chips | Pipeline Method for Compact PIM Designs; Dynamic Duplication Method (DDM); Maximum NN Size Estimation & Deployment in Compact PIM Design |


#### CIM: SRAM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | TCASAI | Purdue | Algorithm Hardware Co-Design for ADC-Less Compute In-Memory Accelerator | reduce ADC overhead in analog CiM architectures; Quantization-Aware Training; Partial Sum Quantization; ADC-Less hybrid analog-digital CiM hardware architecture HCiM |
| 2024 | ISCAS | NYCU | CIMR-V: An End-to-End SRAM-based CIM  Accelerator with RISC-V for AI Edge Device | incorporates CIM layer fusion, convolution/max pooling pipeline, and weight fusion; weight fusion: pipelining the CIM convolution and weight loading |
| 2021 | TCAD | Geogria Tech | DNN+NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for On-Chip Training | non-ideal device properties of NVMS' effect for on-chip training |
| 2020 | ISCAS | JCU | MemTorch: A Simulation Framework for Deep Memristive Cross-Bar Architectures | supports both GPUs and CPUs; integrates directly with PyTorch; simulate non-idealities of memristive devices within cross-bar, tested on VGG-16, CIFAR-10 |
| 2018 | JSSC | MIT | CONV-SRAM: An Energy-Efficient SRAM With In-Memory Dot-Product Computation for Low-Power Convolutional Neural Networks | SRAM-embedded convolution (dot-product) computation architecture for BNN; support multi-bit input-output |
| 2022 | TCAD | NTHU | MARS: Multi-macro Architecture SRAM CIM-Based Accelerator with Co-designed Compressed Neural Networks | sparsity algorithm designed for SRAM CiM; quantization algorithm with BN fusion |
| 2023 | TCAS-I | UIC | MC-CIM: Compute-in-Memory With Monte-Carlo Dropouts for Bayesian Edge Intelligence | SRAM-based CIM macros to accelerate Monte-Carlo dropout; compute reuse between consecutive iterations |
| 2025 | arXiv | Purdue | Hardware-Software Co-Design for Accelerating Transformer Inference Leveraging Compute-in-Memory | SRAM CIM architecture for accelerating attention; optimized for softmax; finer-granularity pipelining strategy |


#### CIM: RRAM

##### RRAM CiM: Simulator
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2018 | TCAD | THU | MNSIM: Simulation Platform for Memristor-Based Neuromorphic Computing System | reference design for largescale neuromorphic accelerator and can also be customized; behavior-level computing accuracy model |
| 2023 | TCAD | THU | MNSIM 2.0: A Behavior-Level Modeling Tool for Processing-In-Memory Architectures | integrated PIM-oriented NN model training and quantization flow; unified PIM memory array model; support for mixed-precision NN operations |
| 2024 | DATE | UCAS | PIMSIM-NN: An ISA-based Simulation Framework for Processing-in-Memory Accelerators | event-driven simulation approach; can evaluate the optimizations of software and hardware independently |


##### RRAM CiM: Architecture


| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | ASPLOS | Purdue & HP | PUMA: A Programmable Ultra-efficient Memristor-based Accelerator for Machine Learning Inference | Programmable and general-purpose ReRAM based ML Accelerator; Supports an instruction set; Has potential for DNN training; Provides simulator that accepts model |
| 2018 | ICRC | Purdue & HP | Hardware-Software Co-Design for an Analog-Digital Accelerator for Machine Learning | compiler to translate model to ISA; ONNX interpreter to support models in common DL frame work; simulator to evaluate performance |
| 2023 | NANOARCH | HUST | Heterogeneous Instruction Set Architecture for RRAM-enabled In-memory Computing | General ISA for RRAM CiM & digital heterogeneous architecture; a tile-processing unit-array three-level architecture |
| 2023 | VLSI | Purdue | X-Former: In-Memory Acceleration of Transformers | in-memory accelerate attention layers; intralayer sequence blocking dataflow; provides a simulator |
| 2024 | VLSI-SoC | RWTH Aachen University | Architecture-Compiler Co-design for ReRAM-Based Multi-core CIM Architectures | inference latency predictions and analysis of the crossbar utilization for CNN |

##### RRAM CiM: Architecture optimization

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | MICRO | HUST | DRCTL: A Disorder-Resistant Computation  Translation Layer Enhancing the Lifetime and  Performance of Memristive CIM Architecture | address conversion method for dynamic scheduling; hierarchical wear-leveling (HWL) strategy for reliability improvement; data layout-aware selective remapping (LASR) to improve communication locality and reduce latency |
| 2024 | DATE | RWTH Aachen University | CLSA-CIM: A Cross-Layer Scheduling Approach for Computing-in-Memory Architectures | algorithm to decide which parts of NN are duplicated to reduce inference latency; cross layer scheduling on tiled CIM architectures |
| 2024 | TC | SJTU | ERA-BS: Boosting the Efficiency of ReRAM-Based  PIM Accelerator With Fine-Grained  Bit-Level Sparsity | bit-level sparsity in both weights and activations; bit-flip scheme; dynamic activation sparsity exploitation scheme |
| 2023 | TETCI | TU Delft | Accurate and Energy-Efficient Bit-Slicing for RRAM-Based Neural Networks | unbalanced bit-slicing scheme for higher accuracy; holistic solution using 2's compliment |

##### RRAM CiM: Modeling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | AICAS | RWTH Aachen University | A Calibratable Model for Fast Energy Estimation of MVM Operations on RRAM Crossbars | system energy model for MVM on ReRAM crossbars; methodology to study the effect of the selection transistor and wire parasitics in 1T1R crossbar arrays |

##### RRAM CiM: Training optimization
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | APIN | SWU | Multi-optimization scheme for in-situ training of memristor neural network based on contrastive learning | optimizations to the deployment method, loss function and gradient calculation; compensation measures for non-ideal effects |
| 2025 | TNNLS | SNU | Efficient Hybrid Training Method for Neuromorphic Hardware Using Analog Nonvolatile Memory | Hybrid offline-online training method |

##### RRAM CiM: Compiler
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | TACO | HUST | A Compilation Tool for Computation Offloading in ReRAM-based CIM Architectures | compilation tool to migrate legacy programs to CPU/CIM heterogeneous architectures; a model to quantify the performance gain |
| 2023 | DAC | CAS | PIMCOMP: A Universal Compilation Framework for Crossbar-based PIM DNN Accelerators | compiler based on Crossbar/IMA/Tile/Chip hierarchy; "low latency" and "high throughput" mode; genetic algorithm to optimize weight replication and core mapping; scheduling algorithms for complex DNN |
| 2024 | ASPLOS | CAS | CIM-MLC: A Multi-level Compilation Stack for Computing-In-Memory Accelerators | compilation stack for various CIM accelerators; multi-level DNN scheduling approach |

##### RRAM CiM: Float-Point processing

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2023 | SC | UCLA | ReFloat: Low-Cost Floating-Point Processing in ReRAM for Accelerating Iterative Linear Solvers | data format and accelerator architecture |
| 2024 | DATE | UESTC | AFPR-CIM: An Analog-Domain Floating-Point RRAM -based Compute- In- Memory Architecture with Dynamic Range Adaptive FP-ADC | all-analog domain CIM architecture for FP8 calculations; adaptive dynamic range FP-ADC & FP-DAC |
| 2025 | arXiv | GWU | A Hybrid-Domain Floating-Point Compute-in-Memory Architecture for Efficient Acceleration of High-Precision Deep Neural Networks | SRAM based hybrid-domain FP CIM architecture; detailed circuit schematics and physical layouts |

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
| 2023 | VLSI-SoC | RWTH Aachen University | Mapping of CNNs on multi-core RRAM-based CIM architectures | architecture optimized for communication; compiler algorithms for conv2D layer; cycle-accurate simulator|
| 2023 | TODAES | UCAS | Mathematical Framework for Optimizing Crossbar Allocation for ReRAM-based CNN Accelerators | formulate a crossbar allocation problem for ReRAM-based CNN accelerators; dynamic programming based solver; models the performance considering allocation problem |

##### RRAM CiM: Transformer

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | TODAES | HUST | A Cascaded ReRAM-based Crossbar Architecture for Transformer Neural Network Acceleration | cascaded crossbar arrays that uses transimpedance amplifiers; data mapping scheme to store signed operands; ADC virtualization scheme |

##### RRAM Non-ideal Effects

Non-ideal Aware Methods: data types, training algiruthm, SRAM for compensation. Refer to [Data Type](#data-type).

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2019 | DATE | Georgia Tech | Design of Reliable DNN Accelerator with Un-reliable ReRAM | dynamical fixed point data representation format; device variation aware training methodology |
| 2020 | DAC | ASU | Accurate Inference with Inaccurate RRAM Devices: Statistical Data, Model Transfer, and On-line Adaptation | introduce statistical variations in knowledge distillation; On-line sparse adaptation with a small SRAM array |
| 2020 | DATE | SJTU | Go Unary: A Novel Synapse Coding and Mapping Scheme for Reliable ReRAM-based Neuromorphic Computing | unary coding; priority mapping* |
| 2022 | TCAD | ASU | Hybrid RRAM/SRAM in-Memory Computing for Robust DNN Acceleration | integrates an RRAM-based IMC macro with a digital SRAM macro using a programmable shifter to compensate for RRAM variations; ensemble learning |
| 2024 | LATS | AMU | Analysis of Conductance Variability in RRAM for  Accurate Neuromorphic Computing | analyzation and quantification of conductance variability in RRAMs; analysis of conductance variation over multiple cycles |
| 2023 | ISCAS | TAMU | Memristor-based Offset Cancellation Technique in Analog Crossbars | peripheral circuitry to remove the systematic offset of crossbar |

#### CIM: Hybrid Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | Science | NTHU | Fusion of memristor and digital compute-in-memory processing for energy-efficient edge computing | Fusion of ReRAM and SRAM CiM; ReRAM SLC & MLC Hybrid; Current quantization; Weight shifting with compensation |
| 2024 | IPDPS | Georgia Tech | Harmonica: Hybrid Accelerator to Overcome Imperfections of Mixed-signal DNN Accelerators | select and transfer imperfectionsensitive weights to digital accelerator; hybrid quantization(weights on analog part is more quantized) |
| 2023 | GLSVLSI | USC | Heterogeneous Integration of In-Memory Analog Computing Architectures with Tensor Processing Units | hybrid TPU-IMAC architecture; TPU for conv, CIM for fc |
| 2025 | ASPLOS | CAS | PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System | dynamic parallelism-aware task scheduling for llm decoding; online kernel characterization for heterogeneous architectures; hybrid PIM units for compute-bound and memory-bound kernels |
| 2025 | arXiv | PKU | Leveraging Compute-in-Memory for Efficient Generative Model Inference in TPUs | Energy-efficient CIM core integration in TPUs (replace the original MXU); CIM-MXU with systolic data path; Array dimension scaling for CIM-MXU;  Area-efficient CIM macro design; Mapping engine for generative model inference |
| 2025 | arXiv | ASU | H3PIMAP: A Heterogeneity-Aware Multi-Objective DNN Mapping Framework on Electronic-Photonic Processing-in-Memory Architectures | Electronic-Photonic-PIM Accelerator; coresponding mapping framework and evaluation infrastructure |
| 2024 | ASP-DAC | Keio | OSA-HCIM: On-The-Fly Saliency-Aware Hybrid SRAM CIM with Dynamic Precision Configuration | On-the-fly Saliency-Aware precision configuration scheme; Hybrid CIM Array for DCIM and ACIM using split-port SRAM |
| 2023 | ICCAD | SJTU | TL-nvSRAM-CIM: Ultra-High-Density Three-Level ReRAM-Assisted Computing-in-nvSRAM with DC-Power Free Restore and Ternary MAC Operations | DCpower-free weight-restore from ReRAM; ternary SRAM-CIM mechanism with differential computing scheme |

#### CIM: Quantization

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | TCAD | BUAA | CIMQ: A Hardware-Efficient Quantization Framework for Computing-In-Memory-Based Neural Network Accelerators | bit-level sparsity induced activation quantization; quantizing partial sums to decrease required resolution of ADCs; arraywise quantization granularity |
| 2018 | CVPR | Google | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | integer-only inference arithmetic; quantizesh both weights and activations as 8-bit integers, bias 32-bit; provides both quantized inference framework and training frame work |
| 2024 | TCAD | BUAA | CIM²PQ: An Arraywise and Hardware-Friendly Mixed Precision Quantization Method for Analog Computing-In-Memory | mixed precision quantization method based on evolutionary algorithm; arraywise quantization granularity; evaluation method to obtain the performance of strategy on the CIM |
| 2023 | ISLPED | Purdue | Partial-Sum Quantization for Near ADC-Less Compute-In-Memory Accelerators | ADC-Less and near ADC-Less CiM accelerators; CiM hardware aware DNN quantization methodology |
| 2023 | ICCD | SJTU | PSQ: An Automatic Search Framework for Data-Free Quantization on PIM-based Architecture | post-training quantization framework without retraining; hardware-aware block reassembly |

#### CIM: Digital CIM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2025 | ISCAS | CAS | StreamDCIM: A Tile-based Streaming Digital CIM Accelerator with Mixed-stationary Cross-forwarding Dataflow for Multimodal Transformer | tile-based reconfigurable CIM macro microarchitecture; mixed-stationary cross-forwarding dataflow; ping-pong-like finegrained compute-rewriting pipeline |

#### NVM

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | ISCAS | UMCP | On-Chip Adaptation for Reducing Mismatch in Analog Non-Volatile Device Based Neural Networks | float-gate transistors based; hot-electron injection to address the issue of mismatch and variation |
| 2023 | DATE | UniBo | End-to-End DNN Inference on a Massively Parallel Analog In Memory Computing Architecture | many-core heterogeneous architecture; general-purpose system based on RISC-V cores and nvAIMC cores; based on Phase-Change Memory(PCM); |

## Photonic computing

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | Nature | SUT | 11 TOPS photonic convolutional accelerator for optical neural networks | universal optical convolutional accelerator for vector processing |


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

##### LLM Accelerators

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | DATE | NTU | ViTA: A Highly Efficient Dataflow and Architecture for Vision Transformers| highly efficient memory-centric dataflow; fused special function module for non-linear functions; A comprehensive DSE of ViTA Kernels and VMUs |
| 2025 | arXiv | SJTU | ROMA: A Read-Only-Memory-based Accelerator for QLoRA-based On-Device LLM | hybrid ROM-SRAM architecture for on-device LLM; "B-ROM" design for area-efficient ROM; fused cell integration of ROM and compute unit; QLoRA rank adaptation for task-specific tuning; on-chip storage optimization for quantized models |

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

#### Fault-Tolerant Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2009 | ICCD | NUS | The Salvage Cache: A fault-tolerant cache architecture for next-generation memory technologies | fault-bit protection for divisions; victim map based division replacement |
| 2011 | CASES | UCSD | FFT-Cache: A Flexible Fault-Tolerant Cache Architecture for Ultra Low Voltage Operation | flexible defect map for faulty block; FDM configuration algorithm; non-functional lines minimization |


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
| 2024 | TCAS | SYSU | CINOC: Computing in Network-On-Chip With Tiled Many-Core Architectures for Large-Scale General Matrix Multiplications | computable input buffers;  thread execution free from fine-grained instruction control; data-aware
thread execution |

##### Topology

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2021 | HPCA | George Washington University | Adapt-NoC: A Flexible Network-on-Chip Design for Heterogeneous Manycore Architectures | mux based adaptable router architecture; adaptable link design; reinforcement learning based subNoC optimization algorithm |
| 2022 | HPCA | Huawei | Application Defined On-chip Networks for Heterogeneous Chiplets: An Implementation Perspective | bufferless multi-ring NoC design; application-architecture-physical co-design method; architecture expressiveness; deadlock resolution SWAP mechanism |
| 2024 | MICRO | Tsinghua University | Ring Road: A Scalable Polar-Coordinate-based 2D Network-on-Chip Architecture | Ring Road topology based on isolated cycles and trees; polar coordinate DOR(dimension-order-routing); inter/intra-chip decouple routing algorithm |
| 2024 | Arxiv | Washington State University | Atleus: Accelerating Transformers on the Edge Enabled by 3D Heterogeneous Manycore Architectures | heterogeneous 3D 
NoC; pipeline design across heterogeneous resources; crossbar-wise quantization |
| 2024 | ISLPED | WSU | HeTraX: Energy Efficient 3D Heterogeneous Manycore Architecture for Transformer Acceleration | 3D integration;  distinct planar tiers where each tier is tailor-made for either MHA or the FF network; alleviate memory bottlenecks while preventing frequent rewrites on ReRAM crossbars |

##### Interconnect

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2012 | SIGCOMM | Carnegie Mellon University | On-Chip Networks from a Networking Perspective: Congestion and Scalability in Many-Core Interconnects | congestion control mechanism for bufferless NoC; interval-based congestion control algorithm; simple injection throttling algorithm |
| 2023 | ICCAD | University of Central Florida | ARIES: Accelerating Distributed Training in Chiplet-based Systems via Flexible Interconnects | directional bypassing link; ARIES link with transistor; ARIES all-reduce optimization algorithm |
| 2023 | MICRO | Tsinghua University | Heterogeneous Die-to-Die Interfaces: Enabling More Flexible Chiplet Interconnection Systems | heterogeneous interface hetero-PHY and hetero-channel; hetero-channel routing algorithm; application-aware scheduling |

##### Processing on NoC

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2017 | ISVLSI | Ruhr-Universität Bochum | Data Stream Processing in Network-on-Chip | data stream processing unit(DSPU); operation mode based DSPU programming framework |  
| 2019 | HPCA | Texas A&M University | Active-Routing: Compute on the Way for Near-Data Processing | active-routing tree; vector processing in cache block for regular access pattern; data prefetch for irregular access pattern |
| 2020 | HPCA | Drexel University | SnackNoC: Processing in the Communication Layer | communication fabric quantification; central packet manager for instruction flit; router compute unit as dataflow pe |

##### Traffic Control

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2017 | ISCA | Texas A&M University | APPROX-NoC: A Data Approximation Framework for Network-On-Chip Architectures | value approximate technique VAXX; encoder/decoder module pair for data compression; approximate value compute logic |
| 2017 | ICCD | HIT | ABDTR: Approximation–Based Dynamic Traffic Regulation for Networks–on–Chip Systems | approximate computing based dynamic traffic regulation technique; lightweight design including controller, throttler and approximater | 
| 2019 | DATE | SCUT | ACDC: An Accuracy- and Congestion-aware Dynamic Traffic Control Method for Networks-on-Chip | quality loss and network congestion modeling; autoregressive model based flow prediction method | 

##### Fault-Tolerant Communication

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2014 | VLSI | ICT | ZoneDefense: A Fault-Tolerant Routing for 2-D Meshes Without Virtual Channels | fault chains based faulty blocks construction; floor/ceiling rule based defense zone forming; L/F chain routing |
| 2017 | TPDS | NTU | Path-Diversity-Aware Fault-Tolerant Routing Algorithm for Network-on-Chip Systems | path diversity analysis; fault-location-based path diversity; PDA-FTR algorithm |
| 2019 | DATE | University of Michigan | SiPterposer: A Fault-Tolerant Substrate for Flexible System-in-Package Design | blowing based customized topology; lightweight ECC module based defect tolerance |
| 2022 | DATE | Colorado State University | DeFT: A Deadlock-Free and Fault-Tolerant Routing Algorithm for 2.5D Chiplet Networks | virtual network based deadlock freedom; congestion-aware vertical link selection |

#### Router

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2016 | HPCA | KTH Royal Institute of Technology | DVFS for NoCs in CMPs: A Thread Voting Approach | thread voting based DVFS machenism; pre-defined region-based V/F adjustment algorithm |
| 2022 | HPCA | Chalmers | FastTrackNoC: A NoC with FastTrack Router Datapaths | non-turning hops; direct FastTrack flit path; zero-load latency analysis |
| 2022 | HPCA | University of Toronto | Stay in your Lane: A NoC with Low-overhead Multi-packet Bypassing | FastFlow flow controll method; time-division-multiplexed (TDM) based non-overlapping FastPass-lanes; FastPass for throughput enhancement |
| 2023 | HPCA | THU | A Scalable Methodology for Designing Efficient Interconnection Network of Chiplets | interface grouping; hypercube construction algorithm; deadlock-free adaptive routing algorithm; safe/unsafe flow control; network interleaving method |
| 2025 | arXiv | SJTU | StreamGrid: Streaming Point Cloud Analytics via Compulsory Splitting and Deterministic Termination | "compulsory splitting" for reducing on-chip buffer size; "deterministic termination" for regularizing non-deterministic operations; line buffer optimization for point cloud pipelines; ILP-based buffer size minimization |

#### RDMA
| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2024 | arXiv | UC Riverside | GPUVM: GPU-driven Unified Virtual Memory | "GPUVM" architecture for on-demand paging; RDMA-capable NIC for GPU memory management; GPU thread-based memory management and page migration; reuse-oriented paged memory for efficient eviction; high-level programming abstraction for GPU memory extension |

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
| 2023 | ISCA | THU | MapZero: Mapping for Coarse-grained Reconfigurable Architectures with Reinforcement Learning and Monte-Carlo Tree Search | GAT based DFG and CGRA embedding; routing penalty based reinforcement learning; Monte-Carlo tree search space exploration |
| 2023 | VLSI | IIT Kharagpur | Application Mapping Onto Manycore Processor Architectures Using Active Search Framework | RNN based active search framework; IP-Core Numbering Scheme; active search with/without pretraining |
| 2024 | HPCA | THU | Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators | layer-centric encoding method; DP-based graph partition algorithm; SA based D2D link communication optimization |
| 2024 | ASPLOS | THU | Cocco: Hardware-Mapping Co-Exploration towards Memory Capacity-Communication Optimization | consumption-centric flow based subgraph execution scheme; main/side region based memory management |

##### Optimization Modeling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | FPGA | ETH Zurich | Flexible Communication Avoiding Matrix Multiplication on FPGA with High-Level Synthesis | computation and I/O decomposition model for matrix multiplication; 1D array collapse mapping method; internal double buffering |
| 2021 | HPCA | Georgia Tech | Heterogeneous Dataflow Accelerators for Multi-DNN Workloads | heterogeneous dataflow accelerators (HDAs) for DNN; dataflow flexibility; high utilization across the sub-accelerators |
| 2023 | MICRO | Alibaba; CUHK | ArchExplorer: Microarchitecture Exploration Via Bottleneck Analysis | dynamic event-dependence graph(EDG); induced DEG based critical path construction; bottleneck-removal-driven DSE |
| 2023 | ISCA | THU | Inter-layer Scheduling Space Definition and Exploration for Tiled Accelerators | inter-layer encoding method; temperal cut; spatial cut; RA tree analysis |

##### Fault Tolerant Mapping

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2017 | SC | NIT | High-performance and energy-efficient fault-tolerance core mapping in NoC | weighted communication energy; placing unmapped vertices region; application core graph; spare core placement algorithm |
| 2019 | IVLSI | UESTC | Optimized mapping algorithm to extend lifetime of both NoC and cores in many-core system | lifetime budget metric; LBC-LBL mapping algorithm; electro-migration fault model |

##### Reliability Management

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | DATE | University of Turku | Thermal-Cycling-aware Dynamic Reliability Management in Many-Core System-on-Chip | Coffin-Mason equation based reliability model; reliability-aware mapping/scheduling; dynamic power management |
| 2024 | Arxiv | TMU | A Two-Level Thermal Cycling-Aware Task Mapping Technique for Reliability Management in Manycore Systems | temperature based bin packing; task-to-bin assignment; thermal cycling-aware based task-to-core mapping |
| 2024 | Arxiv | TMU | A Reinforcement Learning-Based Task Mapping Method to Improve the Reliability of Clustered Manycores | mean time to failure; density-based spatial clustering of applications with noise algorithm |

#### Task Scheduling

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2017 | HPCA | UGent | Reliability-Aware Scheduling on Heterogeneous Multicore Processors | core reliability characteristics difference; system soft error rate; sampling-based reliability-aware scheduling algorithm |
| 2020 | TCAD | ASU | Runtime Task Scheduling Using Imitation Learning for Heterogeneous Many-Core Systems | offline Oracle optimizaion strategy; hierarchical imitation learning based scheduling; two-level scheduling |
| 2023 | ICCAD | PKU | Memory-aware Scheduling for Complex Wired Networks with Iterative Graph Optimization | topology-aware pruning algorithm; integer linear programming scheduling method; sub-graph fusion algorithm ; memory-aware graph partitioning|
| 2023 | MICRO | Duke | Si-Kintsugi: Towards Recovering Golden-Like Performance of Defective Many-Core Spatial Architectures for AI | graph alignment algoithm for dataflow graph and platform pe grap; producer-consumer pattern dataflow generation algorithm |

### Reconfigurable Architecture

### Many-core Architecture

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2015 | HPCA | Cornel | Increasing Multicore System Efficiency through Intelligent Bandwidth Shifting | online bandwidth shifting mechanism; prefetch usefulness (PU) level |
| 2015 | HPCA | IBM | XChange: A Market-based Approach to Scalable Dynamic Multi-resource Allocation in Multicore Architectures | CMP multiresource allocation mechanism XChange; market framework based modeling |
| 2018 | MICRO | SNU | RpStacks-MT: A High-throughput Design Evaluation Methodology for Multi-core Processors | graph-based multi-core performance model; distance-based memory system model; dynamic scheduling reconstruction method |
| 2023 | MICRO | THU | MAICC : A Lightweight Many-core Architecture with In-Cache Computing for Multi-DNN Parallel Inference | slice improved and hardware-implemented reduction CIM; ISA extension for CIM; CNN layer segmentation and mapping algorithm |
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

### RTL Code Generation

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2013 | DAC | Columbia University | A Method to Abstract RTL IP Blocks into C++ Code and Enable High-Level Synthesis | process communication graph; I/O port loop unrolling; HLS design space expansion |
| 2023 | DATE | New York University | Benchmarking Large Language Models for Automated Verilog RTL Code Generation | verilog code training corpus; multi-level verilog coding problems for analysis |
| 2024 | ISEDA | UESTC | GraphRTL: an Agile Design Framework of RTL Code from Data Flow Graphs | graph error detection kernel; DFS based graph equivalent reconstruction; template/scala based DFG and CFG merging |
| 2024 | Arxiv | UCSD | MAGE: A Multi-Agent Engine for Automated RTL Code Generation | multi-agent; high-temperature sampling and ranking; verilog-state checkpoint debugging |

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
| 2020 | MICRO | Georgia Tech; NVIDIA | MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings | data-centric mapping; data reuse analysis; TemperalMap; SpatialMap; analytical cost model |
| 2023 | ISPASS | Georgia Tech | ASTRA-sim2.0: Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale | graph-based training-loop execution; multi-dimensional heterogeneous topology construction; analytical network backend |
| 2024 | ATC | THU | Evaluating Chiplet-based Large-Scale Interconnection Networks via Cycle-Accurate Packet-Parallel Simulation | packet-centric simulation; critical resources recorading for process-order-induced deviations; unimportant stages elimination |
| 2025 | arXiv | UCLM | Understanding Intra-Node Communication in HPC Systems and Datacenters | Intra-/inter-node communication interference; Packet-level simulation (OMNeT++); PCIe/NVLink modeling; LLM communication patterns (DP, TP, PP) impact |

### Performance Analysis



#### Redundancy Detection

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2020 | SC | NC State | ZeroSpy: Exploring Software Inefficiency with Redundant Zeros | code-centric analysis for instruction detection; data-centric analysis for data detection |
| 2020 | SC | NC State | GVPROF: A Value Profiler for GPU-Based Clusters | temporal/spatial load/store redundancy; hierarchical sampling for reducing monitoring overhead; bidirectional search algorithm on dependency graph |
| 2022 | ASPLOS | NC State | ValueExpert: Exploring Value Patterns in GPU-accelerated Applications value-related inefficiencies | data value pattern recoginition; value flow graph; parallel intervals merging algorithm |
| 2022 | SC | NC State | Graph Neural Networks Based Memory Inefficiency Detection Using Selective Sampling | dead store; silent store; silent load; assembly-level procedural control-flow embedding; dynamic value semantic embedding; relative positional encoding for different compilation options |

#### Variation Impact

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2009 | HPCMP | UCSD | Measuring and Understanding Variation in Benchmark Performance | MPI communication variation; distribution of performance variation |
| 2016 | SC | UNM | Understanding Performance Interference in Next-Generation HPC Systems | extreme value theory; bulk-synchronous parallel based modeling; gang/earliest deadline first scheduling |

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

#### Physical Effects

| Year | Venue | Authors | Title | Tags |
|-|-|-|-|-|
| 2004 | ICCAD | UCLA | A thermal-driven floorplanning algorithm for 3D ICs | combined bucket and 2D array; tile stack based model; horizontal and vertical heat flow analysis |
| 2016 | IJHMT | UCR | Analysis of critical thermal issues in 3D integrated circuits | thermal hotspots; impact of thermal interface materials; power distribution; processor pitch and area |
| 2019 | DAC | UCF | Noise Injection Adaption: End-to-End ReRAM Crossbar Non-ideal Effect Adaption for Neural Network Mapping | stuck-at-fault; crossbar wire resistance based IR drop; thermal noise model; shot noise; random telegraph noise |