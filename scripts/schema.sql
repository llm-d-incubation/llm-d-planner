-- PostgreSQL schema for NeuralNav benchmark data
-- This schema matches the real benchmark data structure from integ-oct-29.sql
-- Both synthetic and real data use this exact schema

CREATE TABLE IF NOT EXISTS exported_summaries (
    id uuid NOT NULL,
    config_id text NOT NULL,
    model_hf_repo text NOT NULL,
    provider text,
    type text NOT NULL,
    ttft_mean double precision NOT NULL,
    ttft_p90 double precision NOT NULL,
    ttft_p95 double precision NOT NULL,
    ttft_p99 double precision NOT NULL,
    e2e_mean double precision NOT NULL,
    e2e_p90 double precision NOT NULL,
    e2e_p95 double precision NOT NULL,
    e2e_p99 double precision NOT NULL,
    itl_mean double precision,
    itl_p90 double precision,
    itl_p95 double precision,
    itl_p99 double precision,
    tps_mean double precision,
    tps_p90 double precision,
    tps_p95 double precision,
    tps_p99 double precision,
    hardware text,
    hardware_count integer,
    framework text,
    requests_per_second double precision NOT NULL,
    responses_per_second double precision,
    tokens_per_second double precision NOT NULL,
    mean_input_tokens double precision NOT NULL,
    mean_output_tokens double precision NOT NULL,
    huggingface_prompt_dataset text,
    jbenchmark_created_at timestamp without time zone NOT NULL,
    entrypoint text,
    docker_image text,
    framework_version text,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    loaded_at timestamp without time zone,
    prompt_tokens integer,
    prompt_tokens_stdev integer,
    prompt_tokens_min integer,
    prompt_tokens_max integer,
    output_tokens integer,
    output_tokens_min integer,
    output_tokens_max integer,
    output_tokens_stdev integer,
    profiler_type text,
    profiler_image text,
    profiler_tag text,
    CONSTRAINT exported_summaries_pkey PRIMARY KEY (id)
);

-- Create indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_benchmark_lookup
ON exported_summaries(model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens);

CREATE INDEX IF NOT EXISTS idx_traffic_patterns
ON exported_summaries(prompt_tokens, output_tokens);

CREATE INDEX IF NOT EXISTS idx_model_hardware
ON exported_summaries(model_hf_repo, hardware);

-- Add comments explaining the schema
COMMENT ON TABLE exported_summaries IS 'Benchmark performance data for LLM models with various hardware configurations and traffic patterns';
COMMENT ON COLUMN exported_summaries.id IS 'Unique identifier (UUID)';
COMMENT ON COLUMN exported_summaries.config_id IS 'Configuration hash identifier';
COMMENT ON COLUMN exported_summaries.model_hf_repo IS 'HuggingFace repository path (e.g., meta-llama/Llama-3.1-8B-Instruct)';
COMMENT ON COLUMN exported_summaries.hardware IS 'GPU type (e.g., NVIDIA-L4, NVIDIA-A100-80GB, H100, H200)';
COMMENT ON COLUMN exported_summaries.hardware_count IS 'Number of GPUs (tensor parallel size)';
COMMENT ON COLUMN exported_summaries.prompt_tokens IS 'Target prompt length (integer)';
COMMENT ON COLUMN exported_summaries.output_tokens IS 'Target output length (integer)';
COMMENT ON COLUMN exported_summaries.mean_input_tokens IS 'Actual mean input tokens from benchmark run';
COMMENT ON COLUMN exported_summaries.mean_output_tokens IS 'Actual mean output tokens from benchmark run';
COMMENT ON COLUMN exported_summaries.ttft_p95 IS 'Time to First Token p95 (milliseconds)';
COMMENT ON COLUMN exported_summaries.itl_p95 IS 'Inter-Token Latency p95 (milliseconds per token)';
COMMENT ON COLUMN exported_summaries.e2e_p95 IS 'End-to-End latency p95 (milliseconds)';

-- Create scenario_summary_environment_loads table (from real data schema)
CREATE TABLE IF NOT EXISTS scenario_summary_environment_loads (
    scenario_summary_id uuid NOT NULL,
    environment character varying NOT NULL,
    created_at timestamp(6) without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT scenario_summary_environment_loads_pkey PRIMARY KEY (scenario_summary_id, environment),
    CONSTRAINT scenario_summary_environment_loads_scenario_summary_id_fkey
        FOREIGN KEY (scenario_summary_id)
        REFERENCES exported_summaries(id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);
