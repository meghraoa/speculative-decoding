# Fast Inference from Transformers via Speculative Decoding 

This repository contains our hands-on implementation of transformer optimization techniques, developed as part of the Large Language Models (LLM) course at MVA. The implementation is based on this [paper](https://arxiv.org/abs/2211.17192).

This work was conducted by the following students:

- Abdallah MEGHRAOUI
- Mohamed SAIDI
- Yassine KADDAMI

## Overview

Speculative decoding is a novel technique to reduce inference latency in autoregressive models without compromising output quality. This method leverages two models:
- A **smaller "draft" model** that proposes candidate tokens rapidly.
- A **larger "target" model** that verifies and corrects these proposals efficiently.

By combining speculative execution and parallelized validation, this approach decodes multiple tokens per iteration, dramatically improving throughput for text generation tasks.