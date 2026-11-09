with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

replacements = {
    "?? Biophysical Parameters": "⚙️ Biophysical Parameters",
    "?? Dataset Selection": "📂 Dataset Selection",
    "?? Output Matrix": "📋 Output Matrix",
    "??? Automated Microenvironment Deconvolution": "🔬 Automated Microenvironment Deconvolution",
    "?? Drug Sensitivity Profiling (GDSC2-Trained)": "💊 Drug Sensitivity Profiling (GDSC2-Trained)"
}

for bad, good in replacements.items():
    text = text.replace(bad, good)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("SUCCESS: Sidebar and subheader emojis restored!")
