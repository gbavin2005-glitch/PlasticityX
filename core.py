import numpy as np
import pandas as pd
from scipy.stats import zscore

GENE_SETS = {
    "E": ["CDH1", "EPCAM", "CLDN4", "CLDN7", "OCLN", "DSP"],
    "M": ["VIM", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "FN1"],
    "S": ["PROM1", "ALDH1A1", "SOX2", "POU5F1", "NANOG", "KLF4"],
    "Stroma": ["FAP", "ACTA2", "COL1A1", "COL3A1", "PDGFRB"],
    "Immune_Checkpoints": ["CD274", "PDCD1LG2", "CTLA4", "HAVCR2", "LAG3"]
}

TF_CORE = ["ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "SOX2", "POU5F1", "NANOG"]

DRUG_MODELS = {
    "Paclitaxel (Standard Antimitotic)": {"E_sens": -0.65, "M_sens": 0.85, "Stem_sens": 0.70, "base_ic50": 3.2},
    "Cisplatin (DNA Crosslinker)": {"E_sens": -0.45, "M_sens": 0.60, "Stem_sens": 0.80, "base_ic50": 4.5},
    "RSL3 (Ferroptosis / GPX4 Inhibitor)": {"E_sens": 0.80, "M_sens": -0.85, "Stem_sens": -0.90, "base_ic50": 1.8},
    "Navitoclax / ABT-263 (BCL-XL/BCL-2)": {"E_sens": 0.50, "M_sens": -0.70, "Stem_sens": -0.75, "base_ic50": 2.4},
    "Gefitinib (EGFR TKI)": {"E_sens": -0.80, "M_sens": 0.90, "Stem_sens": 0.55, "base_ic50": 5.1}
}

def calculate_shannon_entropy(matrix: np.ndarray) -> np.ndarray:
    p = np.abs(matrix) / (np.sum(np.abs(matrix), axis=0, keepdims=True) + 1e-9)
    entropy = -np.sum(p * np.log2(p + 1e-9), axis=0)
    denom = np.max(entropy) - np.min(entropy)
    return (entropy - np.min(entropy)) / (denom if denom != 0 else 1.0)

def score_plasticity(df_expr: pd.DataFrame, sigma: float = 1.0, lambda_entropy: float = 0.5, alpha: float = 0.6):
    df = df_expr.copy()
    df.index = [str(x).strip().upper() for x in df.index]
    values = df.to_numpy(dtype=float)
    genes = list(df.index)
    samples = list(df.columns)

    row_means = np.mean(values, axis=1, keepdims=True)
    row_stds = np.std(values, axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    z_matrix = (values - row_means) / row_stds

    gene_to_row = {g: i for i, g in enumerate(genes)}
    e_idx = [gene_to_row[g] for g in GENE_SETS["E"] if g in gene_to_row]
    m_idx = [gene_to_row[g] for g in GENE_SETS["M"] if g in gene_to_row]
    s_idx = [gene_to_row[g] for g in GENE_SETS["S"] if g in gene_to_row]
    stroma_idx = [gene_to_row[g] for g in GENE_SETS["Stroma"] if g in gene_to_row]
    checkpoint_idx = [gene_to_row[g] for g in GENE_SETS["Immune_Checkpoints"] if g in gene_to_row]

    mean_E = np.mean(z_matrix[e_idx, :], axis=0) if e_idx else np.zeros(len(samples))
    mean_M = np.mean(z_matrix[m_idx, :], axis=0) if m_idx else np.zeros(len(samples))
    mean_S = np.mean(z_matrix[s_idx, :], axis=0) if s_idx else np.zeros(len(samples))
    stromal_score = np.mean(z_matrix[stroma_idx, :], axis=0) if stroma_idx else np.zeros(len(samples))
    checkpoint_score = np.mean(z_matrix[checkpoint_idx, :], axis=0) if checkpoint_idx else np.zeros(len(samples))

    stromal_pos = np.clip(stromal_score, a_min=0, a_max=None)
    d_emt = (mean_M - mean_E) / np.sqrt(2)
    diff_sq = (mean_E - mean_M) ** 2
    p_hem = np.exp(-diff_sq / (2 * (sigma ** 2))) * ((mean_E + mean_M) / 2)

    cd44_cd24 = np.zeros(len(samples))
    if "CD44" in gene_to_row and "CD24" in gene_to_row:
        cd44_raw = values[gene_to_row["CD44"], :]
        cd24_raw = values[gene_to_row["CD24"], :]
        ratio = np.log1p(cd44_raw) - np.log1p(cd24_raw)
        r_std = np.std(ratio)
        cd44_cd24 = (ratio - np.mean(ratio)) / (r_std if r_std != 0 else 1.0)
    s_csc = mean_S + cd44_cd24

    entropy_score = calculate_shannon_entropy(z_matrix)
    p_pos = np.clip(p_hem, a_min=0, a_max=None)
    s_pos = np.clip(s_csc, a_min=0, a_max=None)
    euclidean_mis = np.sqrt(p_pos**2 + s_pos**2)
    omri = euclidean_mis * (1.0 + (lambda_entropy * entropy_score))
    tip_score = omri / (1.0 + (alpha * stromal_pos))

    phenotypes = []
    for d, p, s in zip(d_emt, p_hem, s_csc):
        if p > 0.25 and s > 0.0:
            phenotypes.append("High-Risk Hybrid (CSC+)")
        elif d > 0.35:
            phenotypes.append("Mesenchymal")
        elif d < -0.35:
            phenotypes.append("Epithelial")
        else:
            phenotypes.append("Intermediate / Basal")

    results_df = pd.DataFrame({
        "Sample_ID": samples,
        "Directional_EMT": np.round(d_emt, 3),
        "Hybrid_Plasticity": np.round(p_hem, 3),
        "Stemness_Score": np.round(s_csc, 3),
        "Stromal_Contamination": np.round(stromal_score, 3),
        "Immune_Checkpoint_Burden": np.round(checkpoint_score, 3),
        "OMRI_Crude": np.round(omri, 3),
        "TIP_Decontaminated": np.round(tip_score, 3),
        "Phenotype": phenotypes
    })

    drug_predictions = {}
    for drug, weights in DRUG_MODELS.items():
        predicted_ic50 = weights["base_ic50"] + (weights["E_sens"] * mean_E) + (weights["M_sens"] * mean_M) + (weights["Stem_sens"] * s_csc)
        drug_predictions[drug] = np.round(predicted_ic50, 3)
    drug_df = pd.DataFrame(drug_predictions, index=samples)
    z_df = pd.DataFrame(z_matrix, index=genes, columns=samples)
    return results_df, drug_df, z_df
