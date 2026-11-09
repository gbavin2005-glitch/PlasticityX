with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace corrupted question-mark prefixes with proper Unicode emojis
replacements = {
    '?? PlasticityX:': '🧬 PlasticityX:',
    '?? 2D Landscape': '📊 2D Landscape',
    '??? TME & Drug Sensitivity': '💊 TME & Drug Sensitivity',
    '?? Survival (KM)': '📈 Survival (KM)',
    '?? 3D Waddington & TF': '🏔️ 3D Waddington & TF',
    '?? Validation & Benchmarks': '🎯 Validation & Benchmarks',
    '?? Clinical Survival Stratification': '📈 Clinical Survival Stratification',
    '?? 3D Waddington Landscape': '🏔️ 3D Waddington Landscape',
    '?? Transcription Factor Driver Cascade': '🧬 Transcription Factor Driver Cascade',
    '?? Validation Benchmarks': '🎯 Validation Benchmarks',
    '?? Performance Scoreboard': '🏆 Performance Scoreboard',
    '? **Statistically Validated:**': '✅ **Statistically Validated:**',
    '? **Statistically Significant:**': '✅ **Statistically Significant:**'
}

for bad, good in replacements.items():
    text = text.replace(bad, good)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("SUCCESS: All emoji titles restored cleanly in UTF-8!")
