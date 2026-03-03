# Proposal: Contribute NeuralNav to SIG Benchmarking and integrate with Config Explorer

## Summary

Contributors from the [Config Explorer](https://github.com/llm-d/llm-d-benchmark/tree/main/config_explorer) and [NeuralNav](https://github.com/redhat-et/neuralnav) teams have been collaborating and have identified strong technical synergies between the two projects. We'd like to propose contributing NeuralNav to the SIG Benchmarking ecosystem and working toward integration with Config Explorer.

This is a joint collaboration proposal. Both projects are fully functional, independently valuable applications today. We envision them starting as independent peers within a shared repository and progressively integrating over time -- sharing components, aligning data formats, and eventually converging into a unified planning tool for llm-d deployments. A detailed proposal with integration architecture and roadmap is available in [Proposal: An llm-d-planner for rapid llm-d configuration planning](https://docs.google.com/document/d/1jnsWjjjxyVr1SjVVaZz5i378KHS6xx6HCNqHAPo3q8w/edit?usp=sharing).

## Background

**Config Explorer** helps users analyze GPU memory requirements, estimate performance using BentoML's roofline model, and explore llm-d benchmark results -- including P/D disaggregation configurations and scheduler parameter analysis. It answers: *"Can this model run on this GPU, and what performance can I expect?"*

**NeuralNav** guides users from business requirements to production deployments. Users describe their use case in natural language, and NeuralNav extracts requirements, maps them to SLO targets and traffic profiles, queries benchmark data, ranks model+GPU configurations across multiple dimensions (accuracy, cost, latency, complexity), generates Kubernetes manifests, and deploys. It answers: *"Given my business needs, what should I deploy and how do I get it running?"*

Both projects are working applications with Streamlit frontends, CLIs, and Python library APIs. Both help users interpret benchmark data and plan deployments -- but from different angles and with different capabilities.

## Why the projects complement each other

We've done a detailed integration analysis and found that the two projects have remarkably little overlap in their core logic:

**Config Explorer provides capabilities NeuralNav lacks:**
- GPU memory estimation (weights, KV cache, activation) with attention-type awareness (MHA/GQA/MQA/MLA)
- Synthetic performance estimation via roofline model for any model on any GPU
- P/D disaggregation data model and llm-d scheduler parameter analysis
- Benchmark sweep visualization with Pareto-front analysis

**NeuralNav provides capabilities Config Explorer lacks:**
- Natural language intent extraction (use case description to structured requirements)
- Use case to SLO mapping
- Model quality scoring
- Multi-criteria ranking across 4 dimensions with 5 ranked views
- YAML generation (KServe/vLLM/HPA/ServiceMonitor)
- Replica and scaling calculations

The overlap is limited to two shared concepts -- SLO compliance filtering and GPU cost lookups -- both of which would benefit from a single source of truth.

A few specific integration opportunities stand out:

1. **Expanded coverage**: NeuralNav can only recommend configurations that have been empirically benchmarked. Config Explorer's roofline model could fill gaps with synthetic estimates, presented alongside real benchmark data with tiered confidence levels.

2. **Quality-aware ranking at scale**: Config Explorer surfaces many (model, GPU) configurations but doesn't evaluate whether the model is good at the user's task. NeuralNav's multi-criteria scoring ensures expanded coverage doesn't come at the cost of recommendation quality.

3. **P/D disaggregation**: NeuralNav currently supports only aggregated vLLM deployments. Config Explorer already models P/D disaggregation natively in its data schema -- this could become the foundation for llm-d-native deployment guidance rather than something that needs to be built from scratch.

4. **Benchmark feedback loop**: NeuralNav's recommendations could identify high-value (model, GPU) combinations that lack empirical data, helping prioritize future benchmark runs.

## Proposed collaboration

We'd like to propose the following:

1. **Contribute NeuralNav** to the SIG Benchmarking ecosystem as a complete working project.

2. **Enable the two projects to leverage each other's technology** over time, starting with the areas of strongest synergy (benchmark data access, synthetic performance estimation, capacity planning).

3. **Gradually extract shared components** as collaboration matures -- for example, benchmark data formats and loading, hardware/model specification logic, performance estimation, and visualization utilities.

Initially, both projects would continue to function as independent applications -- contributing NeuralNav doesn't break existing functionality. Over time, as shared components are extracted and integration deepens, the two projects would progressively converge into a unified tool.

## Proposed structure

We propose to **create a new repository** (name TBD) under SIG Benchmarking to host both Config Explorer and NeuralNav. The reasoning:

- **Keep llm-d-benchmark focused** on benchmarking infrastructure and benchmark results -- what it does well today.
- **Give applications their own home** -- Config Explorer and NeuralNav are both user-facing applications that help people *use* benchmark data. A dedicated repository would be a natural home for tools in this category.
- **Enable independent evolution** -- both projects can continue developing while integration work proceeds alongside them.
- **Facilitate shared component extraction** -- as common patterns emerge (data access, hardware specs, cost calculations), they can be factored out within the same repository.

## Next steps / discussion

We'd welcome feedback from SIG maintainers and the broader community on:

- **Interest**: Is this kind of collaboration something the SIG would like to pursue?
- **Repository structure**: New repo vs. hosting within llm-d-benchmark -- what makes sense for the project?
- **Integration priorities**: Which integration points are most valuable to the community?

We'd be happy to demo NeuralNav at an upcoming SIG meeting and answer any questions about the proposal.
