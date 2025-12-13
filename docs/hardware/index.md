# Hardware

For each paper, identify its primary contribution and assign it to the single most appropriate category below.

## Processor Microarchitecture

Focuses on a single processing core and its components. Includes:

* Instruction set design
* Branch prediction
* Other core-level components and techniques

## Parallel and Multi-Processor Architecture

Covers systems with multiple processing units and their interactions. Includes:

* Multi-core processor architecture
* Many-core processor architecture
* GPU architecture
* Cache coherence protocols
* Memory consistency models
* System-level integration techniques such as:
    * Chiplets
    * 3D stacking

## Memory Architecture

Concerns memory subsystems and their interactions. Includes:

* Memory hierarchy design
* 存算
    * Processing-in-Memory (PIM)
    * Processing-Near-Memory (PNM)
    * Computation-in-Memory (CIM)

## Interconnection Networks

Addresses the communication fabric. Includes:

* Network-on-Chip (NoC) - topology, routing, flow control
* Note: Focus should be on the network **itself**, not primarily on **using** it to build a parallel architecture. Most CXL papers likely belong in other categories unless the primary contribution is the CXL protocol/architecture itself.

## Domain-Specific Accelerators

Concerns accelerators tailored for specific applications. Includes:

* Reconfigurable architectures like FPGA/CGRA
* Accelerators for domains such as AI, graph processing, bioinformatics, etc.

## Security and Reliability

Covers hardware mechanisms for trust and resilience. Includes:

* Side-channel countermeasures (hardware aspects)
* Fault detection/mitigation hardware

## Emerging Technologies

Explores architectures based on novel technologies or paradigms. Includes:

* Quantum computing architectures
* Photonic computing architectures
* Note: CIM should be grouped under Memory Architecture.
