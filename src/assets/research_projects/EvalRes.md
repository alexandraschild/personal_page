## Overview

Vision–language models (VLMs) have achieved impressive results on a wide range of visual reasoning and question-answering tasks. Yet they still face significant limitations when handling **high-resolution images** or regions containing small but critical details. Most current architectures, including Vision Transformers (ViTs), are constrained by fixed input resolutions, which can result in significant loss of fine-grained visual information. This limitation not only affects performance on tasks requiring detailed reasoning but also undermines evaluations of higher-level abilities such as chain-of-thought reasoning.

Our research aims to **systematically understand, characterize and benchmark** these limitations and their reasons, providing tools and metrics to assess how well VLMs scale with image resolution, aspect ratio, and visual detail. By disentangling failures caused by architectural constraints from other reasoning abilities, we can better understand the true capabilities of current models and guide the design of next-generation VLMs.

## Motivation

High-resolution and detail-rich image processing is critical across many domains:
 
- **Diagrams, charts, and large technical plans**, where small elements carry essential meaning. 
- **Text-heavy or structured visual content**, such as tables and flowcharts. 
- **Medical and satellite imaging**, where critical information may occupy a tiny portion of the image. 

Existing VLMs often downsample or crop images to fit fixed-resolution inputs, discarding small but essential visual cues. While some recent models support larger resolutions or adaptive tokenization, **even state-of-the-art architectures struggle with small details**, non-standard aspect ratios, and high-resolution content.

## Research Goals

1. Develop a **systematic framework** to extend any VQA benchmark for evaluating performance on small details, higher resolutions, and varying aspect ratios.  
2. Conduct **robust experimental evaluations** across multiple widely used VLMs and benchmarks to categorize failure modes and measure the impact of architectural bottlenecks.  
3. Propose **novel metrics** for quantifying a model’s ability to scale to high-resolution inputs and handle non-standard image dimensions.
4. Propose **novel architectures** that enable robustness of VLMs to High-Resolution and Detail-Rich Visual Inputs and scale natually to longer visual context. 

## Current Approach

Our evaluation methodology involves:

- Controlled experiments that vary image resolution, aspect ratio, and size of critical details independently.  
- Benchmarking multiple commonly used VLMs across a range of visual reasoning tasks.  
- Quantifying performance degradation due to image scaling separately from other reasoning abilities, allowing fairer and more interpretable comparisons.  

We also explore architectural bottlenecks that cause these issues.

## Broader Impact and Outlook

By systematically analyzing VLMs' high-resolution limitations, this research provides a foundation for:

- **Designing next-generation VLMs** capable of fine-grained reasoning over large, complex visual inputs.  
- **Benchmarking and evaluating** models more rigorously in domains where small details are crucial.  
- **Enhancing applications** in various domains, such as document understanding, architecture and engineering, medical and satelite imaging, where failure to capture small visual cues can have significant consequences.

Our work helps pave the way for **high-fidelity visual understanding** in multimodal AI, enabling models that can reason accurately over both detailed and large-scale visual content.

<h2><span style="color:#4A90E2;">Collaborations</span></h2>

If you read till here, there is a chance you are interested in similar topics or a joint project:) I am interested in collaborations and joint research projects related to this one. In particular, I welcome opportunities to work with researchers exploring geometric deep learning, vision–language models, spatial and geometric reasoning. Feel free to drop me a message if you are interested in joint work.