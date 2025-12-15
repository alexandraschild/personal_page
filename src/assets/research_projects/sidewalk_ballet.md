## Overview

Many regions of interest in real-world images are not defined by a single object category, but by **semantically complex concepts** such as social groups, activities, or functional zones. Detecting these regions requires reasoning about relationships, proximity, geometry, and higher-order structure—capabilities that large language models (LLMs) and vision–language models (VLMs increasingly demonstrate at a conceptual level. However, despite their strong reasoning abilities, current VLMs remain **poorly grounded**: they struggle to translate semantic understanding into precise, pixel-level localization such as bounding boxes or region masks.

This research direction investigates how to bridge this gap between high-level semantic reasoning and low-level visual grounding, with the goal of enabling VLMs to **holistically detect regions defined by complex semantics** directly from natural language descriptions.

## Background and Motivation

This work is motivated by challenges encountered in the *Sidewalk Ballet* project, a collaboration with MIT Urban Planning aimed at understanding group-level social interactions in public spaces at scale. Such interactions are central to urban design, as they provide insights into how public environments foster social vibrancy, inclusion, and collective behavior.

Crucially, socially interacting groups are not visually defined by explicit markers. Instead, they emerge from **subtle visual cues** such as interpersonal distance, orientation, co-movement, and relational structure. These signals go far beyond traditional object detection and require interpreting **semantically rich and relational patterns** across multiple individuals.

## Decompositional Approach: AAAI Phase

In the first phase of this research, presented at AAAI AI for Social Impact, we addressed the problem through a **decompositional pipeline** tailored to the urban planning use case. The approach broke down semantically complex region detection into three stages:

1. **Individual Detection** – identifying single people using a standard human detector.
2. **Pairwise Semantic Reasoning** – prompting a VLM to reason about relationships between all pairs of detected individuals (e.g., whether two people are socially interacting).
3. **Aggregation** – combining pairwise predictions into higher-order group structures that define socially interacting regions.

While this pipeline demonstrated that VLMs can successfully reason about social relationships when appropriately scaffolded, it also exposed significant limitations. The process is computationally expensive, sensitive to compounding errors at each stage, and fundamentally *indirect*: semantic regions are inferred only after extensive post-processing rather than being grounded natively in the image.

## Limitations of Current VLMs

The need for such pipelines highlights a deeper issue: current VLMs lack mechanisms for **direct pixel-wise semantic grounding**. Although they can describe complex relationships in language, they struggle to output spatial representations—such as bounding boxes or masks—that are tightly aligned with the image. This suggests architectural and representational mismatches between visual encoders, language reasoning modules, and spatial outputs.

Understanding *why* VLMs fail at this task is a central question of the current research phase. Is the limitation rooted in visual tokenization, cross-modal alignment, training objectives, or the absence of explicit spatial supervision? Addressing these questions is essential for moving beyond task-specific pipelines toward general solutions.

## Toward Holistic Region Detection

The current direction of this project explores **holistic architectures** that enable VLMs to directly predict semantically complex regions from text prompts, without intermediate decomposition into objects and relations. Inspired by recent work on unified vision–language segmentation and detection, we investigate model designs


<h2><span style="color:#4A90E2;">Collaborations</span></h2>

If you read till here, there is a chance you are interested in similar topics or a joint project:) 

I am interested in collaborations and joint research projects related to this direction. In particular, I welcome opportunities to work with researchers exploring vision–language models, spatial and geometric reasoning, multimodal grounding. Feel free to drop me a message if you are interested in contributing to this project or joint work on similar topics.

