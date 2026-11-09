with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

start_marker = "    patient_x = results[\"Directional_EMT\"].values"
end_marker = "    st.plotly_chart(fig_radar, width=\"stretch\")"

if start_marker in text and end_marker in text:
    pre = text.split(start_marker)[0]
    post = text.split(end_marker)[1]

    new_tab4 = """    # Phenotype color palette
    color_map = {
        "High-Risk Hybrid (CSC+)": "#e74c3c",
        "Mesenchymal-like": "#9b59b6",
        "Epithelial-like": "#1abc9c",
        "Intermediate": "#f39c12"
    }

    # Plot sample points grouped by phenotype for interactive legend & tooltips
    for pheno, grp in results.groupby("Phenotype"):
        p_x = grp["Directional_EMT"].values
        p_y = grp["Stemness_Score"].values
        p_z = 0.8 * ((p_x**2) - 1.2)**2 - 1.2 * np.exp(-((p_x**2) + (p_y - 1.0)**2) / (2 * (sigma**2))) + 0.25 * (p_y**2) + 0.12
        hover_txt = [
            f"<b>Sample:</b> {sid}<br><b>Phenotype:</b> {pheno}<br><b>TIP:</b> {tip:.3f}<br><b>EMT:</b> {emt:.2f}<br><b>Stemness:</b> {stem:.2f}"
            for sid, tip, emt, stem in zip(grp["Sample_ID"], grp["TIP_Decontaminated"], p_x, p_y)
        ]
        fig_3d.add_trace(go.Scatter3d(
            x=p_x, y=p_y, z=p_z,
            mode="markers",
            marker=dict(size=4.5, color=color_map.get(pheno, "#3498db"), opacity=0.95),
            name=pheno,
            text=hover_txt,
            hoverinfo="text"
        ))

    fig_3d.update_layout(
        title="Quasi-Potential Energy Topology: U(Directional EMT, Stemness)",
        scene=dict(
            xaxis_title="Directional EMT (d_EMT)",
            yaxis_title="Stemness Score (s_CSC)",
            zaxis_title="Potential U(x, y)",
            camera=dict(eye=dict(x=1.65, y=-1.65, z=1.25))
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.02, y=0.95)
    )
    st.plotly_chart(fig_3d, width="stretch")

    st.markdown("---")
    st.subheader("🧬 Transcription Factor Driver Cascade")

    selected_sample = st.selectbox("Select Sample ID for TF Profiling:", results["Sample_ID"].tolist())
    col_tf1, col_tf2 = st.columns([1, 1])

    available_tfs = [tf for tf in TF_CORE if tf in z_expr.index]

    if available_tfs:
        tf_loadings = z_expr.loc[available_tfs, selected_sample].sort_values(ascending=True)

        with col_tf1:
            fig_bar = px.bar(
                x=tf_loadings.values,
                y=tf_loadings.index,
                orientation='h',
                color=tf_loadings.values,
                color_continuous_scale="RdBu_r",
                labels={"x": "Z-Score Relative Expression", "y": "Regulator"},
                title=f"Core TF Activation: `{selected_sample}`"
            )
            fig_bar.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_bar, width="stretch")

        with col_tf2:
            radar_cats = list(tf_loadings.index) + [tf_loadings.index[0]]
            radar_vals = list(tf_loadings.values) + [tf_loadings.values[0]]
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_vals,
                theta=radar_cats,
                fill='toself',
                line=dict(color="#e74c3c", width=2),
                fillcolor="rgba(231, 76, 60, 0.3)"
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[min(min(tf_loadings.values) - 0.5, -2.0), max(max(tf_loadings.values) + 0.5, 2.0)]
                    )
                ),
                title=f"Regulatory Fingerprint (`{selected_sample}`)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_radar, width="stretch")"""

    with open("app.py", "w", encoding="utf-8") as f:
        f.write(pre + new_tab4 + post)
    print("SUCCESS: Tab 4 updated with phenotype colors, tooltips, and radar baseline!")
else:
    print("Marker mismatch in Tab 4.")
