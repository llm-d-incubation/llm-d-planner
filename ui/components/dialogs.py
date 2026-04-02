"""Dialog components for the Planner UI.

Winner details, category exploration, and full table dialogs.
"""

import math

import streamlit as st
from helpers import format_display_name, format_gpu_config, format_use_case_name, get_scores


def render_score_bar(label: str, icon: str, score: float, bar_class: str, contribution: float):
    """Render a score with progress bar using native Streamlit."""
    if isinstance(score, float) and math.isnan(score):
        score = 50
    if isinstance(contribution, float) and math.isnan(contribution):
        contribution = 0
    score = max(0, min(100, score))
    col_label, col_bar, col_val = st.columns([2, 4, 2])
    with col_label:
        st.write(f"**{label}**")
    with col_bar:
        st.progress(int(score))
    with col_val:
        st.write(f"{score:.0f} (+{contribution:.1f})")


def _render_winner_details(winner: dict, priority: str, extraction: dict):
    """Render detailed winner information inside the dialog."""

    # Handle both backend format (scores) and UI format (score_breakdown)
    backend_scores = winner.get("scores", {}) or {}
    ui_breakdown = winner.get("score_breakdown", {}) or {}
    breakdown = {
        "quality_score": backend_scores.get("accuracy_score", ui_breakdown.get("quality_score", 0)),
        "latency_score": backend_scores.get("latency_score", ui_breakdown.get("latency_score", 0)),
        "cost_score": backend_scores.get("price_score", ui_breakdown.get("cost_score", 0)),
        "capacity_score": backend_scores.get(
            "complexity_score", ui_breakdown.get("capacity_score", 0)
        ),
    }

    # === FINAL RECOMMENDATION BOX ===
    st.subheader("Final Recommendation")

    model_name = winner.get("model_name", "Unknown Model")
    gpu_config = winner.get("gpu_config", {}) or {}
    hardware = gpu_config.get("gpu_type", winner.get("hardware", "H100"))
    hw_count = gpu_config.get("gpu_count", winner.get("hardware_count", 1))
    tp = gpu_config.get("tensor_parallel", 1)
    replicas = gpu_config.get("replicas", 1)

    backend_scores = winner.get("scores", {}) or {}
    final_score = backend_scores.get("balanced_score", winner.get("final_score", 0))
    quality_score = breakdown.get("quality_score", 0)

    ttft_p95 = winner.get("predicted_ttft_p95_ms", 0)
    itl_p95 = winner.get("predicted_itl_p95_ms", 0)
    e2e_p95 = winner.get("predicted_e2e_p95_ms", 0)
    throughput_qps = winner.get("predicted_throughput_qps", 0)

    if not ttft_p95:
        benchmark_slo = winner.get("benchmark_slo", {})
        slo_actual = benchmark_slo.get("slo_actual", {}) if benchmark_slo else {}
        throughput_data = benchmark_slo.get("throughput", {}) if benchmark_slo else {}
        ttft_p95 = slo_actual.get("ttft_p95_ms", slo_actual.get("ttft_mean_ms", 0))
        itl_p95 = slo_actual.get("itl_p95_ms", slo_actual.get("itl_mean_ms", 0))
        e2e_p95 = slo_actual.get("e2e_p95_ms", slo_actual.get("e2e_mean_ms", 0))
        throughput_qps = (
            throughput_data.get("tokens_per_sec", 0) / 100
            if throughput_data.get("tokens_per_sec")
            else 0
        )

    ttft_display = f"{int(ttft_p95)}" if ttft_p95 and ttft_p95 > 0 else "—"
    itl_display = f"{int(itl_p95)}" if itl_p95 and itl_p95 > 0 else "—"
    e2e_display = f"{int(e2e_p95)}" if e2e_p95 and e2e_p95 > 0 else "—"
    max_rps = f"{throughput_qps:.1f}" if throughput_qps and throughput_qps > 0 else "—"

    priority_text = priority.replace("_", " ").title()

    hw_display = f"{hw_count}x {hardware}"
    if tp > 1 and replicas > 1:
        hw_display += f" (TP={tp}, R={replicas})"

    rec_html = f"""<div style="padding: 2rem; border-radius: 1.25rem; margin-bottom: 1.5rem; ">
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; padding-bottom: 1rem; ">
        <span style="font-size: 2.5rem;"></span>
        <div>
            <h2 style="margin: 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;">RECOMMENDATION</h2>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem;">Based on {priority_text} optimization</p>
                    </div>
                    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
        <div style="display: flex; flex-direction: column; gap: 1.25rem;">
            <div style="padding: 1rem; border-radius: 0.75rem; ">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Model</p>
                <p style="margin: 0; font-size: 1.25rem; font-weight: 700;">{model_name}</p>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem;">Quality Score: <span style="font-weight: 700;">{quality_score:.0f}%</span></p>
                    </div>
            <div style="padding: 1rem; border-radius: 0.75rem; ">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Hardware Configuration</p>
                <p style="margin: 0; font-size: 1.25rem; font-weight: 700;">{hw_display}</p>
                    </div>
                </div>
        <div style="padding: 1.25rem; border-radius: 0.75rem; ">
            <p style="margin: 0 0 1rem 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">Expected SLO (p95)</p>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; ">
                    <span style="font-size: 0.85rem;">Max RPS</span>
                    <span style="font-weight: 800; font-size: 1.1rem;">{max_rps}</span>
            </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; ">
                    <span style="font-size: 0.85rem;">TTFT (p95)</span>
                    <span style="font-weight: 800; font-size: 1.1rem;">{ttft_display}<span style="font-size: 0.7rem; "> ms</span></span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; ">
                    <span style="font-size: 0.85rem;">ITL (p95)</span>
                    <span style="font-weight: 800; font-size: 1.1rem;">{itl_display}<span style="font-size: 0.7rem; "> ms</span></span>
                    </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0;">
                    <span style="font-size: 0.85rem;">E2E (p95)</span>
                    <span style="font-weight: 800; font-size: 1.1rem;">{e2e_display}<span style="font-size: 0.7rem; "> ms</span></span>
                    </div>
                    </div>
                    </div>
                </div>
    <div style="margin-top: 1.5rem; padding-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
        <div style="font-size: 0.85rem;"><strong >Verified</strong> - Real benchmark data</div>
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 0.9rem;">Final Score:</span>
            <span style="padding: 0.5rem 1.25rem; border-radius: 0.5rem; font-weight: 900; font-size: 1.5rem; ">{final_score:.1f}</span>
            </div>
        </div>
</div>"""

    st.markdown(rec_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Winner Details: Score Breakdown")

    st.markdown(
        """
    <div style="padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; ">
        <p style="margin: 0; font-size: 0.9rem;">
            <strong >Score Format:</strong> <code style="padding: 0.1rem 0.4rem; border-radius: 0.25rem;">87 → +17.4</code> means the model scored <strong>87/100</strong> in this category, contributing <strong >+17.4 points</strong> to the final weighted score.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            f'<h3 style="font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem; ">{winner.get("model_name", "Unknown")}</h3>',
            unsafe_allow_html=True,
        )

        w_accuracy = st.session_state.get("weight_accuracy", 5)
        w_latency = st.session_state.get("weight_latency", 2)
        w_cost = st.session_state.get("weight_cost", 4)
        w_total = w_accuracy + w_latency + w_cost
        weights = {
            "accuracy": w_accuracy / w_total if w_total > 0 else 0.33,
            "latency": w_latency / w_total if w_total > 0 else 0.33,
            "cost": w_cost / w_total if w_total > 0 else 0.33,
        }

        q_score = breakdown.get("quality_score", 0)
        l_score = breakdown.get("latency_score", 0)
        c_score = breakdown.get("cost_score", 0)
        cap_score = breakdown.get("capacity_score", 0)

        q_contrib = q_score * weights["accuracy"]
        l_contrib = l_score * weights["latency"]
        c_contrib = c_score * weights["cost"]
        cap_contrib = 0  # Complexity weight is 0 in UI

        render_score_bar("Accuracy", "", q_score, "score-bar-accuracy", q_contrib)
        render_score_bar("Latency", "", l_score, "score-bar-latency", l_contrib)
        render_score_bar("Cost", "", c_score, "score-bar-cost", c_contrib)
        render_score_bar("Capacity", "", cap_score, "score-bar-capacity", cap_contrib)

    with col2:
        st.subheader("Why This Model?")

        model_name = winner.get("model_name", "Unknown")
        use_case = extraction.get("use_case", "chatbot_conversational")
        use_case_display = format_use_case_name(use_case)

        st.markdown(
            f"""
        <div style="padding: 1rem; border-radius: 0.75rem; margin-bottom: 1rem; ">
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.6;">
                <strong>{model_name}</strong> ranked highest for <strong>{use_case_display}</strong>
                with a balanced score of <strong>{breakdown.get('quality_score', 0):.0f}</strong> accuracy,
                <strong>{breakdown.get('latency_score', 0):.0f}</strong> latency,
                and <strong>{breakdown.get('cost_score', 0):.0f}</strong> cost efficiency.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        pros = winner.get("pros", ["Top Quality", "Fast Responses"])
        cons = winner.get("cons", [])

        st.markdown(
            '<p style="font-weight: 600; margin-bottom: 0.5rem;">Strengths:</p>',
            unsafe_allow_html=True,
        )
        pros_html = (
            '<div style="display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 1rem;">'
        )
        for pro in pros:
            pros_html += f'<span class="tag tag-pro">{pro}</span>'
        pros_html += "</div>"
        st.markdown(pros_html, unsafe_allow_html=True)

        if cons:
            st.markdown(
                '<p style="font-weight: 600; margin-bottom: 0.5rem;">Trade-offs:</p>',
                unsafe_allow_html=True,
            )
            cons_html = '<div style="display: flex; flex-direction: column; gap: 0.4rem;">'
            for con in cons:
                cons_html += f'<span class="tag tag-con">{con}</span>'
            cons_html += "</div>"
            st.markdown(cons_html, unsafe_allow_html=True)

        st.markdown(
            '<hr style="border-color: rgba(212, 175, 55, 0.3); margin: 1rem 0;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
        <p ><strong >Final Score:</strong> <code style="padding: 0.35rem 0.75rem; border-radius: 0.25rem; font-weight: 700; font-size: 1.1rem;">{winner.get('final_score', 0):.1f}/100</code></p>
        <p style="color: rgba(212, 175, 55, 0.8); font-style: italic;">Based on {priority.replace('_', ' ').title()} priority weighting</p>
        """,
            unsafe_allow_html=True,
        )

    # Display benchmark SLO data
    benchmark_slo = winner.get("benchmark_slo", {}) or {}
    gpu_config = winner.get("gpu_config", {}) or {}

    ttft_p95_val = winner.get("predicted_ttft_p95_ms") or benchmark_slo.get("slo_actual", {}).get(
        "ttft_p95_ms", 0
    )
    itl_p95_val = winner.get("predicted_itl_p95_ms") or benchmark_slo.get("slo_actual", {}).get(
        "itl_p95_ms", 0
    )
    e2e_p95_val = winner.get("predicted_e2e_p95_ms") or benchmark_slo.get("slo_actual", {}).get(
        "e2e_p95_ms", 0
    )
    throughput_qps_val = winner.get("predicted_throughput_qps") or (
        benchmark_slo.get("throughput", {}).get("tokens_per_sec", 0) / 100
        if benchmark_slo.get("throughput", {}).get("tokens_per_sec")
        else 0
    )

    traffic_profile = winner.get("traffic_profile", {}) or {}
    prompt_tokens_val = traffic_profile.get(
        "prompt_tokens", benchmark_slo.get("token_config", {}).get("prompt", 512)
    )
    output_tokens_val = traffic_profile.get(
        "output_tokens", benchmark_slo.get("token_config", {}).get("output", 256)
    )

    hw_type_val = gpu_config.get("gpu_type", benchmark_slo.get("hardware", "H100"))
    hw_count_val = gpu_config.get("gpu_count", benchmark_slo.get("hardware_count", 1))

    if ttft_p95_val or itl_p95_val or e2e_p95_val:
        st.markdown("---")
        st.subheader("Real Benchmark SLOs (Actual Achievable Performance)")

        st.markdown(
            f"""
        <div style="padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; ">
            <p style="margin: 0; font-size: 0.9rem;">
                <strong >Benchmarks:</strong> Real measured values from vLLM simulation.
                Hardware: <strong >{hw_count_val}x {hw_type_val}</strong> |
                Token Config: <strong >{prompt_tokens_val}→{output_tokens_val}</strong>
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        slo_actual = benchmark_slo.get("slo_actual", {})
        throughput = benchmark_slo.get("throughput", {})

        col1, col2, col3 = st.columns(3)

        ttft_p95_show = ttft_p95_val or slo_actual.get("ttft_p95_ms", 0)
        itl_p95_show = itl_p95_val or slo_actual.get("itl_p95_ms", 0)
        e2e_p95_show = e2e_p95_val or slo_actual.get("e2e_p95_ms", 0)
        tps_show = (
            throughput_qps_val * 100 if throughput_qps_val else throughput.get("tokens_per_sec", 0)
        )

        with col1:
            st.markdown(
                f"""
            <div style="padding: 1.25rem; border-radius: 0.75rem; ">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">⏱️</span>
                    <span style="font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">TTFT</span>
                </div>
                <div style="text-align: center;">
                    <p style="font-weight: 800; font-size: 2rem; margin: 0;">{int(ttft_p95_show) if ttft_p95_show else 'N/A'}<span style="font-size: 1rem; ">ms</span></p>
                    <p style="font-size: 0.75rem; margin: 0.25rem 0 0 0;">p95 latency</p>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div style="padding: 1.25rem; border-radius: 0.75rem; ">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1rem; font-weight: 700;">TTFT</span>
                    <span style="font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">ITL</span>
                </div>
                <div style="text-align: center;">
                    <p style="font-weight: 800; font-size: 2rem; margin: 0;">{int(itl_p95_show) if itl_p95_show else 'N/A'}<span style="font-size: 1rem; ">ms</span></p>
                    <p style="font-size: 0.75rem; margin: 0.25rem 0 0 0;">inter-token latency</p>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div style="padding: 1.25rem; border-radius: 0.75rem; ">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">🏁</span>
                    <span style="font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">E2E</span>
                </div>
                <div style="text-align: center;">
                    <p style="font-weight: 800; font-size: 2rem; margin: 0;">{int(e2e_p95_show) if e2e_p95_show else 'N/A'}<span style="font-size: 1rem; ">ms</span></p>
                    <p style="font-size: 0.75rem; margin: 0.25rem 0 0 0;">end-to-end</p>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="padding: 1rem; border-radius: 0.75rem; text-align: center; ">
                <span style="font-size: 1.25rem;">🚀</span>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Throughput</p>
                <p style="font-weight: 800; font-size: 1.5rem; margin: 0;">{int(tps_show) if tps_show else 'N/A'} <span style="font-size: 0.8rem;">tok/s</span></p>
            </div>
            <div style="padding: 1rem; border-radius: 0.75rem; text-align: center; ">
                <span style="font-size: 1rem; font-weight: 700;">HW</span>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Hardware</p>
                <p style="font-weight: 800; font-size: 1.25rem; margin: 0;">{hw_count_val}x {hw_type_val}</p>
            </div>
            <div style="padding: 1rem; border-radius: 0.75rem; text-align: center; ">
                <span style="font-size: 1rem; font-weight: 700;">Tokens</span>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Token Config</p>
                <p style="font-weight: 700; font-size: 1rem; margin: 0;">{prompt_tokens_val} → {output_tokens_val}</p>
            </div>
        </div>

        <div style="margin-top: 1rem; padding: 0.75rem; border-radius: 0.5rem; ">
            <p style="margin: 0; font-size: 0.8rem; text-align: center;">
                <strong >Data Source:</strong> vLLM Simulation Benchmarks |
                <strong >Model:</strong> {winner.get('model_name', 'Unknown')}
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


@st.dialog("Winner Details", width="large")
def show_winner_details_dialog():
    """Show winner details in a modal dialog."""
    winner = st.session_state.get("balanced_winner") or st.session_state.get(
        "winner_recommendation"
    )
    priority = st.session_state.get("winner_priority", "balanced")
    extraction = st.session_state.get("winner_extraction", {})

    if not winner:
        st.warning("No winner data available.")
        return

    _render_winner_details(winner, priority, extraction)

    if st.button("Close", key="close_dialog_btn", use_container_width=True):
        st.session_state.show_winner_dialog = False
        st.rerun()


@st.dialog("Top 5 Models", width="large")
def show_category_dialog():
    """Show top 5 models for a category with performance data."""
    category = st.session_state.get("explore_category", "balanced")
    use_case = st.session_state.get("detected_use_case", "chatbot_conversational")

    category_config = {
        "balanced": {"title": "Balanced - Top 5", "field": "final", "top5_key": "top5_balanced"},
        "accuracy": {
            "title": "Best Accuracy - Top 5",
            "field": "accuracy",
            "top5_key": "top5_accuracy",
        },
        "latency": {
            "title": "Best Latency - Top 5",
            "field": "latency",
            "top5_key": "top5_latency",
        },
        "cost": {"title": "Best Cost - Top 5", "field": "cost", "top5_key": "top5_cost"},
    }

    config = category_config.get(category, category_config["balanced"])
    top5_list = st.session_state.get(config["top5_key"], [])

    st.subheader(config["title"])
    st.caption(f"Use case: {format_use_case_name(use_case)}")

    if not top5_list:
        st.warning("No models available for this category.")
        if st.button("Close", key="close_cat_dialog_empty"):
            st.session_state.show_category_dialog = False
            st.rerun()
        return

    for i, rec in enumerate(top5_list):
        scores = get_scores(rec)
        model_name = format_display_name(rec.get("model_name", "Unknown"))
        gpu_cfg = rec.get("gpu_config", {}) or {}
        hw_type = gpu_cfg.get("gpu_type", rec.get("hardware", "H100"))
        hw_count = gpu_cfg.get("gpu_count", rec.get("hardware_count", 1))
        hw_display = f"{hw_count}x {hw_type}"
        tp = gpu_cfg.get("tensor_parallel", 1)

        selected_percentile = st.session_state.get("slo_percentile", "p95")
        percentile_suffix = selected_percentile
        percentile_label = selected_percentile.upper() if selected_percentile != "mean" else "Mean"

        benchmark_metrics = rec.get("benchmark_metrics", {}) or {}

        ttft = benchmark_metrics.get(
            f"ttft_{percentile_suffix}", rec.get("predicted_ttft_p95_ms", "N/A")
        )
        itl = benchmark_metrics.get(
            f"itl_{percentile_suffix}", rec.get("predicted_itl_p95_ms", "N/A")
        )
        e2e = benchmark_metrics.get(
            f"e2e_{percentile_suffix}", rec.get("predicted_e2e_p95_ms", "N/A")
        )
        tps = benchmark_metrics.get(f"tps_{percentile_suffix}", 0)

        throughput_display = f"{tps:.0f} tok/s" if tps and tps > 0 else "N/A"
        highlight_score = scores.get(config["field"], 0)

        st.markdown(
            f"""
        <div style="border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-weight: 800; padding: 0.25rem 0.6rem;
                                border-radius: 6px; font-size: 0.85rem;">#{i+1}</span>
                    <div>
                        <div style="font-weight: 700; font-size: 1.05rem;">{model_name}</div>
                        <div style="font-size: 0.8rem;">{hw_display} (TP={tp})</div>
        </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.75rem; font-weight: 800;">{highlight_score:.0f}</div>
                    <div style="font-size: 0.65rem; text-transform: uppercase;">FINAL</div>
                </div>
            </div>
            <div style="display: flex; gap: 1rem; margin-bottom: 0.75rem; padding: 0.5rem;
                        border-radius: 8px;">
                <span style="font-size: 0.8rem;">Acc: {scores['accuracy']:.0f}</span>
                <span style="font-size: 0.8rem;">Lat: {scores['latency']:.0f}</span>
                <span style="font-size: 0.8rem;">Cost: {scores['cost']:.0f}</span>
                <span style="font-size: 0.8rem; font-weight: 600;">Final: {scores['final']:.1f}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem;">
                <div style="padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.65rem;">TTFT ({percentile_label})</div>
                    <div style="font-weight: 700; font-size: 0.9rem;">{ttft if isinstance(ttft, str) else f'{ttft:.0f}ms'}</div>
                </div>
                <div style="padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.65rem;">ITL ({percentile_label})</div>
                    <div style="font-weight: 700; font-size: 0.9rem;">{itl if isinstance(itl, str) else f'{itl:.0f}ms'}</div>
                </div>
                <div style="padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.65rem;">E2E ({percentile_label})</div>
                    <div style="font-weight: 700; font-size: 0.9rem;">{e2e if isinstance(e2e, str) else f'{e2e:.0f}ms'}</div>
                </div>
                <div style="padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.65rem;">Throughput</div>
                    <div style="font-weight: 700; font-size: 0.9rem;">{throughput_display}</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("Close", key="close_cat_dialog", use_container_width=True):
        st.session_state.show_category_dialog = False
        st.rerun()


@st.dialog("All Recommendation Options", width="large")
def show_full_table_dialog():
    """Show the full recommendations table with all categories."""
    ranked_response = st.session_state.get("ranked_response")

    if not ranked_response:
        st.warning("No recommendations available. Please run the recommendation process first.")
        if st.button("Close", key="close_full_table_empty"):
            st.session_state.show_full_table_dialog = False
            st.rerun()
        return

    st.subheader("Configuration Options")
    st.caption("All viable deployment configurations ranked by category")

    total_configs = ranked_response.get("total_configs_evaluated", 0)
    configs_after_filters = ranked_response.get("configs_after_filters", 0)

    st.markdown(
        f'<div style="margin-bottom: 1rem; font-size: 0.9rem;">Evaluated <span style="font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="font-weight: 600;">{configs_after_filters}</span> unique options</div>',
        unsafe_allow_html=True,
    )

    categories = [
        ("balanced", "Balanced"),
        ("best_accuracy", "Best Accuracy"),
        ("lowest_cost", "Lowest Cost"),
        ("lowest_latency", "Lowest Latency"),
    ]

    all_rows = []
    for cat_key, cat_name in categories:
        recs = ranked_response.get(cat_key, [])
        if not recs:
            all_rows.append(
                f'<tr ><td style="padding: 0.75rem 0.5rem;"><span style="font-weight: 600;">{cat_name}</span></td><td colspan="7" style="padding: 0.75rem 0.5rem; font-style: italic;">No configurations found</td></tr>'
            )
        else:
            for i, rec in enumerate(recs[:5]):
                model_name = format_display_name(rec.get("model_name", "Unknown"))
                gpu_str = format_gpu_config(rec.get("gpu_config", {}))
                ttft = rec.get("predicted_ttft_p95_ms", 0)
                cost = rec.get("cost_per_month_usd", 0)
                scores = rec.get("scores", {}) or {}
                accuracy = scores.get("accuracy_score", 0)
                balanced = scores.get("balanced_score", 0)
                meets_slo = rec.get("meets_slo", False)
                slo_icon = "Yes" if meets_slo else "No"

                cat_display = (
                    f'<span style="font-weight: 600;">{cat_name}</span> (+{len(recs)-1})'
                    if i == 0
                    else ""
                )

                row = f'<tr ><td style="padding: 0.75rem 0.5rem;">{cat_display}</td><td style="padding: 0.75rem 0.5rem; font-weight: 500;">{model_name}</td><td style="padding: 0.75rem 0.5rem; font-size: 0.85rem;">{gpu_str}</td><td style="padding: 0.75rem 0.5rem; text-align: right; ">{ttft:.0f}ms</td><td style="padding: 0.75rem 0.5rem; text-align: right; ">${cost:,.0f}</td><td style="padding: 0.75rem 0.5rem; text-align: center; ">{accuracy:.0f}</td><td style="padding: 0.75rem 0.5rem; text-align: center; ">{balanced:.1f}</td><td style="padding: 0.75rem 0.5rem; text-align: center;">{slo_icon}</td></tr>'
                all_rows.append(row)

    header = '<thead><tr ><th style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Category</th><th style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Model</th><th style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">GPU Config</th><th style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">TTFT</th><th style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Cost/mo</th><th style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Acc</th><th style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Score</th><th style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">SLO</th></tr></thead>'

    table_html = f'<table style="width: 100%; border-collapse: collapse; border-radius: 8px;">{header}<tbody>{"".join(all_rows)}</tbody></table>'

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Close", key="close_full_table_dialog", use_container_width=True):
        st.session_state.show_full_table_dialog = False
        st.rerun()
