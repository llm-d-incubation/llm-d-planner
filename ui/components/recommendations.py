"""Recommendation display components for the Planner UI.

Category cards, top-5 table, options list, and recommendation results.
"""

from typing import Any

import streamlit as st
from api_client import deploy_and_generate_yaml
from helpers import format_display_name, format_gpu_config, get_scores


def _render_filter_summary():
    """Render the SLO filter summary stats."""
    ranked_response = st.session_state.get("ranked_response", {})
    total_configs = ranked_response.get("total_configs_evaluated", 0)
    passed_configs = ranked_response.get("configs_after_filters", 0)

    if total_configs <= 0:
        return

    all_passed = []
    for cat in ["balanced", "best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
        all_passed.extend(ranked_response.get(cat, []))
    unique_models = len({r.get("model_name", "") for r in all_passed if r.get("model_name")})

    filter_pct = passed_configs / total_configs * 100
    st.markdown(
        f"""
    <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1rem;">
        <span style="font-size: 0.85rem;">
            <strong style="color: #10B981;">{passed_configs:,}</strong> configs passed SLO filter
            from <strong>{total_configs:,}</strong> total
            <span >({filter_pct:.0f}% match)</span>
        </span>
        <span >|</span>
        <span style="font-size: 0.85rem;">
            <strong>{unique_models}</strong> unique models
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def _render_category_card(title, recs_list, highlight_field, category_key, col):
    """Render a recommendation card for a category with prev/next navigation."""
    if not recs_list:
        return

    idx_key = f"cat_idx_{category_key}"
    idx = st.session_state.get(idx_key, 0)
    idx = min(idx, len(recs_list) - 1)  # Clamp if list shrank

    rec = recs_list[idx]
    scores = get_scores(rec)
    model_name = format_display_name(rec.get("model_name", "Unknown"))
    gpu_cfg = rec.get("gpu_config", {}) or {}
    hw_type = gpu_cfg.get("gpu_type", rec.get("hardware", "H100"))
    hw_count = gpu_cfg.get("gpu_count", rec.get("hardware_count", 1))
    replicas = gpu_cfg.get("replicas", 1)
    cost = rec.get("cost_per_month_usd", 0)

    # Confidence badge
    bench_metrics = rec.get("benchmark_metrics", {}) or {}
    confidence = bench_metrics.get("confidence_level", "benchmarked")
    badge_styles = {
        "benchmarked": ("Benchmarked", "#10B981", "Based on real hardware benchmark data."),
        "estimated": (
            "Estimated",
            "#F59E0B",
            "Based on roofline model estimation. Actual performance may vary.",
        ),
    }
    badge_label, badge_color, badge_tooltip = badge_styles.get(
        confidence, badge_styles["benchmarked"]
    )
    badge_html = (
        f'<span title="{badge_tooltip}" style="font-size: 0.7rem; font-weight: 600; '
        f"color: {badge_color}; border: 1px solid {badge_color}; border-radius: 4px; "
        f'padding: 1px 6px; margin-left: 8px; vertical-align: middle;">'
        f"{badge_label}</span>"
    )

    # Source badge (e.g., "blis", "llm-optimizer")
    source = bench_metrics.get("source")
    if source:
        badge_html += (
            f'<span title="Data source: {source}" style="font-size: 0.7rem; font-weight: 600; '
            f"color: #6B7280; border: 1px solid #D1D5DB; border-radius: 4px; "
            f'padding: 1px 6px; margin-left: 4px; vertical-align: middle;">'
            f"{source}</span>"
        )

    # Performance metrics (P95)
    ttft = rec.get("predicted_ttft_p95_ms") or 0
    itl = rec.get("predicted_itl_p95_ms") or 0
    e2e = rec.get("predicted_e2e_p95_ms") or 0
    throughput = rec.get("predicted_throughput_qps") or 0

    # Build scores line with highlight on the matching category
    score_items = [
        ("accuracy", "Accuracy", scores["accuracy"]),
        ("cost", "Cost", scores["cost"]),
        ("latency", "Latency", scores["latency"]),
        ("final", "Balanced", scores["final"]),
    ]
    score_parts = []
    for field, label, value in score_items:
        if field == highlight_field:
            score_parts.append(
                f'<span style="color: #1f77b4; font-weight: 700;">{label}: {value:.0f}</span>'
            )
        else:
            score_parts.append(f"{label}: {value:.0f}")
    scores_line = " | ".join(score_parts)

    # Build metrics line
    metrics_line = (
        f"TTFT: {ttft:,}ms | ITL: {itl}ms | E2E: {e2e:,}ms | "
        f"Throughput: {throughput:.1f} rps | Cost: ${cost:,.0f}/mo"
    )

    with col, st.container(border=True):
        st.markdown(
            f'<div style="line-height: 1.7; margin-bottom: 1rem;">'
            f'<strong style="font-size: 1.05rem;">{title}</strong>{badge_html}<br>'
            f'<span style="font-size: 0.9rem;"><strong>Solution:</strong> Model: {model_name} | Hardware: {hw_count}x {hw_type} | Replicas: {replicas}</span><br>'
            f'<span style="font-size: 0.9rem;"><strong>Scores:</strong> {scores_line}</span><br>'
            f'<span style="font-size: 0.9rem;"><strong>Values:</strong> {metrics_line}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Prev/Next navigation (circular) - compact centered layout
        if len(recs_list) > 1:
            last = len(recs_list) - 1

            # Use narrow columns for compact centered navigation
            spacer_l, nav_area, spacer_r = st.columns([0.3, 0.4, 0.3])
            with nav_area:
                btn_prev, counter, btn_next = st.columns([1, 1.5, 1])
                with btn_prev:
                    # Right-align the prev button using nested columns
                    empty, btn = st.columns([0.3, 0.7])
                    with btn:
                        if st.button("◀", key=f"prev_{category_key}"):
                            st.session_state[idx_key] = last if idx == 0 else idx - 1
                            st.session_state.deployment_selected_config = None
                            st.session_state.deployment_selected_category = None
                            st.session_state.deployment_yaml_generated = False
                            st.session_state.deployment_yaml_files = {}
                            st.session_state.deployment_id = None
                            st.session_state.deployment_error = None
                            st.session_state.deployed_to_cluster = False
                            st.rerun()
                with counter:
                    st.markdown(
                        f"<div style='text-align: center; line-height: 2.4; font-size: 0.85rem;'>#{idx + 1} of {len(recs_list)}</div>",
                        unsafe_allow_html=True,
                    )
                with btn_next:
                    # Left-align but add spacing with nested columns
                    btn, empty = st.columns([0.7, 0.3])
                    with btn:
                        if st.button("▶", key=f"next_{category_key}"):
                            st.session_state[idx_key] = 0 if idx == last else idx + 1
                            st.session_state.deployment_selected_config = None
                            st.session_state.deployment_selected_category = None
                            st.session_state.deployment_yaml_generated = False
                            st.session_state.deployment_yaml_files = {}
                            st.session_state.deployment_id = None
                            st.session_state.deployment_error = None
                            st.session_state.deployed_to_cluster = False
                            st.rerun()

        selected_category = st.session_state.get("deployment_selected_category")
        is_selected = selected_category == category_key

        if is_selected:
            # Show custom green "Selected" indicator (visual only)
            st.markdown(
                """
                <div style="
                    width: 100%;
                    padding: 0.5rem 1rem;
                    background-color: #D1FAE5;
                    color: #065F46;
                    border: 1px solid #10B981;
                    border-radius: 0.375rem;
                    font-size: 1rem;
                    font-weight: 400;
                    text-align: center;
                    margin-bottom: 0.5rem;
                ">
                    ✓ Selected
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if st.button("Select", key=f"select_{category_key}", use_container_width=True):
                st.session_state.deployment_selected_config = rec
                st.session_state.deployment_selected_category = category_key
                st.session_state.deployment_yaml_generated = False
                st.session_state.deployment_yaml_files = {}
                st.session_state.deployment_id = None
                st.session_state.deployed_to_cluster = False

                result = deploy_and_generate_yaml(rec)
                if result and result.get("success"):
                    st.session_state.deployment_id = result["deployment_id"]
                    st.session_state.deployment_yaml_files = result["files"]
                    st.session_state.deployment_yaml_generated = True
                else:
                    st.session_state.deployment_yaml_generated = False
                st.rerun()


def render_top5_table(recommendations: list, priority: str):
    """Render Top 5 recommendation cards with filtering summary.

    Uses the backend's pre-ranked lists (ACCURACY-FIRST strategy in analyzer.py).
    """
    st.subheader("Best Model Recommendations")
    _render_filter_summary()

    use_case = st.session_state.get("detected_use_case", "chatbot_conversational")

    if not recommendations:
        st.info("No models available. Please check your requirements.")
        return

    ranked_response = st.session_state.get("ranked_response", {})
    top5_balanced = ranked_response.get("balanced", [])[:5]
    top5_accuracy = ranked_response.get("best_accuracy", [])[:5]
    top5_latency = ranked_response.get("lowest_latency", [])[:5]
    top5_cost = ranked_response.get("lowest_cost", [])[:5]

    st.session_state.top5_balanced = top5_balanced
    st.session_state.top5_accuracy = top5_accuracy
    st.session_state.top5_latency = top5_latency
    st.session_state.top5_cost = top5_cost
    st.session_state.top5_simplest = ranked_response.get("simplest", [])[:5]

    # Render 4 category cards in a 2x2 grid
    col1, col2 = st.columns(2)
    _render_category_card("Balanced", top5_balanced, "final", "balanced", col1)
    _render_category_card("Best Accuracy", top5_accuracy, "accuracy", "accuracy", col2)

    col3, col4 = st.columns(2)
    _render_category_card("Best Latency", top5_latency, "latency", "latency", col3)
    _render_category_card("Best Cost", top5_cost, "cost", "cost", col4)

    total_available = len(recommendations)
    if total_available <= 2:
        use_case_display = use_case.replace("_", " ").title() if use_case else "this task"
        st.info(f"Only {total_available} model(s) have benchmarks for {use_case_display}")


def render_options_list_inline():
    """Render the expanded options list content inline on the page."""
    ranked_response = st.session_state.get("ranked_response")

    if not ranked_response:
        st.warning("No recommendations available. Please run the recommendation process first.")
        return

    st.markdown(
        '<div style="padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; margin-top: 1rem;"><h2 style="margin: 0; font-size: 1.5rem;">Configuration Options</h2><p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">All viable deployment configurations ranked by category</p></div>',
        unsafe_allow_html=True,
    )

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

    table_data = []
    for cat_key, cat_name in categories:
        recs = ranked_response.get(cat_key, [])
        for rec in recs[:5]:
            model_name = format_display_name(rec.get("model_name", "Unknown"))
            gpu_str = format_gpu_config(rec.get("gpu_config", {}))
            ttft = rec.get("predicted_ttft_p95_ms", 0)
            cost = rec.get("cost_per_month_usd", 0)
            scores = rec.get("scores", {}) or {}
            accuracy = scores.get("accuracy_score", 0)
            balanced = scores.get("balanced_score", 0)
            meets_slo = rec.get("meets_slo", False)
            slo_value = 1 if meets_slo else 0

            bench_m = rec.get("benchmark_metrics", {}) or {}
            conf_level = bench_m.get("confidence_level", "benchmarked")

            table_data.append(
                {
                    "category": cat_name,
                    "model": model_name,
                    "gpu_config": gpu_str,
                    "ttft": ttft,
                    "cost": cost,
                    "accuracy": accuracy,
                    "balanced": balanced,
                    "slo": "Yes" if meets_slo else "No",
                    "slo_value": slo_value,
                    "confidence": conf_level,
                }
            )

    if table_data:
        rows_html = []
        for row in table_data:
            rows_html.append(f"""
                <tr
                    data-category="{row["category"]}"
                    data-model="{row["model"]}"
                    data-gpu="{row["gpu_config"]}"
                    data-ttft="{row["ttft"]}"
                    data-cost="{row["cost"]}"
                    data-accuracy="{row["accuracy"]}"
                    data-balanced="{row["balanced"]}"
                    data-slo="{row["slo_value"]}">
                    <td style="padding: 0.75rem 0.5rem; ">{row["category"]}</td>
                    <td style="padding: 0.75rem 0.5rem; font-weight: 500;">{row["model"]}</td>
                    <td style="padding: 0.75rem 0.5rem; font-size: 0.85rem;">{row["gpu_config"]}</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: right; ">{row["ttft"]:.0f}ms</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: right; ">${row["cost"]:,.0f}</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: center; ">{row["accuracy"]:.0f}</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: center; ">{row["balanced"]:.1f}</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: center;">{row["slo"]}</td>
                </tr>
            """)

        table_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }}
            .sortable-table {{
                font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }}
            .sortable-table th {{
                cursor: pointer;
                user-select: none;
                position: relative;
            }}
            .sortable-table th:hover {{
                !important;
            }}
            .sortable-table th.sort-asc::after {{
                content: " ▲";
                font-size: 0.7em;
                }}
            .sortable-table th.sort-desc::after {{
                content: " ▼";
                font-size: 0.7em;
                }}
            .sortable-table tbody tr:hover {{
                !important;
            }}
        </style>
        </head>
        <body>
        <table class="sortable-table" id="recsTableInline" style="width: 100%; border-collapse: collapse; border-radius: 8px;">
            <thead>
                <tr >
                    <th onclick="sortTableInline(0, 'string')" style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Category</th>
                    <th onclick="sortTableInline(1, 'string')" style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Model</th>
                    <th onclick="sortTableInline(2, 'string')" style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">GPU Config</th>
                    <th onclick="sortTableInline(3, 'number')" style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">TTFT</th>
                    <th onclick="sortTableInline(4, 'number')" style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Cost/mo</th>
                    <th onclick="sortTableInline(5, 'number')" style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Acc</th>
                    <th onclick="sortTableInline(6, 'number')" style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Score</th>
                    <th onclick="sortTableInline(7, 'number')" style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">SLO</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
        <script>
            let sortDirectionInline = {{}};

            function sortTableInline(columnIndex, type) {{
                const table = document.getElementById('recsTableInline');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const headers = table.querySelectorAll('th');

                const key = columnIndex;
                sortDirectionInline[key] = sortDirectionInline[key] === 'asc' ? 'desc' : 'asc';
                const isAsc = sortDirectionInline[key] === 'asc';

                headers.forEach(h => {{
                    h.classList.remove('sort-asc', 'sort-desc');
                }});

                headers[columnIndex].classList.add(isAsc ? 'sort-asc' : 'sort-desc');

                rows.sort((a, b) => {{
                    let aVal, bVal;

                    if (type === 'number') {{
                        const attrs = ['category', 'model', 'gpu', 'ttft', 'cost', 'accuracy', 'balanced', 'slo'];
                        const attr = 'data-' + attrs[columnIndex];
                        aVal = parseFloat(a.getAttribute(attr)) || 0;
                        bVal = parseFloat(b.getAttribute(attr)) || 0;
                    }} else {{
                        aVal = a.cells[columnIndex].textContent.trim();
                        bVal = b.cells[columnIndex].textContent.trim();
                    }}

                    if (aVal < bVal) return isAsc ? -1 : 1;
                    if (aVal > bVal) return isAsc ? 1 : -1;
                    return 0;
                }});

                rows.forEach(row => tbody.appendChild(row));
            }}
        </script>
        </body>
        </html>
        """

        st.iframe(table_html, height=450)
        st.markdown(
            '<p style="font-size: 0.85rem; margin-top: 0.5rem;">Click on column headers to sort the table</p>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("No configurations to display")


def render_recommendation_result(result: dict, priority: str, extraction: dict):
    """Render recommendation results with Top 5 table."""

    if extraction is None:
        extraction = {}

    ranked_response = result

    if ranked_response:
        st.session_state.ranked_response = ranked_response

        balanced_recs = ranked_response.get("balanced", [])
        if balanced_recs:
            winner = balanced_recs[0]
            recommendations = balanced_recs
        else:
            for cat in ["best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
                if ranked_response.get(cat):
                    winner = ranked_response[cat][0]
                    recommendations = ranked_response[cat]
                    break
            else:
                st.warning("No recommendations found.")
                return
    else:
        st.warning(
            "Could not fetch ranked recommendations from backend. Ensure the backend is running."
        )
        st.session_state.ranked_response = None
        recommendations = result.get("recommendations", [])
        if not recommendations:
            st.warning("No recommendations found. Try adjusting your requirements.")
            return
        winner = recommendations[0]

    st.session_state.winner_recommendation = winner
    st.session_state.winner_priority = priority
    st.session_state.winner_extraction = extraction

    # Display estimation warnings if any
    estimation_warnings = ranked_response.get("warnings", []) if ranked_response else []
    if estimation_warnings:
        with st.expander(f"Estimation warnings ({len(estimation_warnings)})", expanded=False):
            for warn in estimation_warnings:
                st.warning(warn)

    # Get all recommendations for the cards
    all_recs: list[dict[str, Any]] = []
    for cat in ["balanced", "best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
        cat_recs = (
            st.session_state.ranked_response.get(cat, [])
            if st.session_state.ranked_response
            else []
        )
        all_recs.extend(cat_recs)

    # Remove duplicates by model+hardware
    seen = set()
    unique_recs = []
    for rec in all_recs:
        model = rec.get("model_name", "")
        gpu_cfg = rec.get("gpu_config", {}) or {}
        hw = f"{gpu_cfg.get('gpu_type', 'H100')}x{gpu_cfg.get('gpu_count', 1)}"
        key = f"{model}_{hw}"
        if key not in seen:
            seen.add(key)
            unique_recs.append(rec)

    if unique_recs:
        render_top5_table(unique_recs, priority)

    # === VIEW DEPLOYMENT CONFIGURATION BUTTON ===
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    has_selection = st.session_state.get("deployment_selected_category") is not None

    # Use columns to center and limit button width to ~500px
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        if has_selection:
            if st.button(
                "View Deployment Configuration",
                key="view_deployment_config_btn",
                use_container_width=True,
                type="primary"
            ):
                # Navigate to Deployment tab (index 3)
                st.session_state["_pending_tab"] = 3
                st.rerun()
        else:
            st.button(
                "View Deployment Configuration",
                key="view_deployment_config_btn_disabled",
                use_container_width=True,
                disabled=True
            )

    # === CONFIGURATION OPTIONS SECTION ===
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("Configuration Options", expanded=False):
        render_options_list_inline()
