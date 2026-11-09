with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Locate from fig_km = go.Figure() down to with col_km2:
split_start = "        fig_km = go.Figure()"
split_end = "    with col_km2:"

if split_start in text and split_end in text:
    pre = text.split(split_start)[0] + split_start + "\n\n"
    post = split_end + text.split(split_end)[1]

    km_block = """        if group_mask.sum() > 0 and (~group_mask).sum() > 0:
            kmf_a.fit(durations=merged.loc[group_mask, time_col], event_observed=merged.loc[group_mask, status_col], label=label_a)
            sf_a = kmf_a.survival_function_
            fig_km.add_trace(go.Scatter(
                x=sf_a.index,
                y=sf_a.iloc[:, 0],
                mode="lines",
                name=label_a,
                line=dict(color="#e74c3c", width=3, shape="hv")
            ))

            kmf_b.fit(durations=merged.loc[~group_mask, time_col], event_observed=merged.loc[~group_mask, status_col], label=label_b)
            sf_b = kmf_b.survival_function_
            fig_km.add_trace(go.Scatter(
                x=sf_b.index,
                y=sf_b.iloc[:, 0],
                mode="lines",
                name=label_b,
                line=dict(color="#2ecc71", width=3, shape="hv")
            ))

            lr = logrank_test(
                merged.loc[group_mask, time_col], merged.loc[~group_mask, time_col],
                merged.loc[group_mask, status_col], merged.loc[~group_mask, status_col]
            )
            p_val = lr.p_value

            fig_km.update_layout(
                title=f"Kaplan-Meier Survival Curves (Log-Rank p = {p_val:.4e})",
                xaxis_title="Overall Survival (Months)",
                yaxis_title="Survival Probability",
                template="plotly_dark",
                legend=dict(x=0.02, y=0.15)
            )
            st.plotly_chart(fig_km, width="stretch")\n\n"""

    with open("app.py", "w", encoding="utf-8") as f:
        f.write(pre + km_block + post)
    print("SUCCESS: Tab 3 cleanly rebuilt!")
else:
    print("Could not find markers; let's inspect the exact lines.")
