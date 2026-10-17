import ast
from pathlib import Path

p = Path("app.py")
src = p.read_text(encoding="utf-8-sig")

old1 = "version = config.get(" + chr(34) + "model_version" + chr(34) + ", " + chr(34) + "baseline-v1" + chr(34) + ")"
new1 = old1 + "\nuncertainty_threshold = float(config.get(" + chr(34) + "uncertainty_threshold" + chr(34) + ", 0.60))"
assert old1 in src, "version anchor missing"
src = src.replace(old1, new1, 1)

old2 = (
    "        st.subheader(f" + chr(34) + "Prediction: {best}" + chr(34) + ")\n"
    "        st.metric(" + chr(34) + "Confidence" + chr(34) + ", f" + chr(34) + "{conf * 100:.1f}%" + chr(34) + ")"
)
new2 = (
    "        ranked = sorted(result.items(), key=lambda t: -t[1])\n"
    "        runner_up = ranked[1] if len(ranked) > 1 else (None, 0.0)\n"
    "        spread = conf - runner_up[1]\n"
    "        uncertain = conf < uncertainty_threshold\n"
    "\n"
    "        if uncertain:\n"
    "            st.warning(" + chr(34) + "Low confidence - the model cannot classify this film reliably. "
    "Radiologist review is recommended. The full distribution is shown below for transparency." + chr(34) + ")\n"
    "        st.subheader(f" + chr(34) + "Prediction: {best}" + chr(34) + ")\n"
    "        st.metric(" + chr(34) + "Confidence" + chr(34) + ", f" + chr(34) + "{conf * 100:.1f}%" + chr(34) + ")"
)
assert old2 in src, "prediction block anchor missing"
src = src.replace(old2, new2, 1)

old3 = (
    "        with st.expander(" + chr(34) + "Raw output (JSON)" + chr(34) + "):\n"
    "            st.json(result)"
)
new3 = (
    "        with st.expander(" + chr(34) + "Raw output (JSON)" + chr(34) + "):\n"
    "            st.json(result)\n"
    "\n"
    "        st.subheader(" + chr(34) + "Findings summary (auto-generated)" + chr(34) + ")\n"
    "        summary = [\n"
    "            f" + chr(34) + "The model assigns **{conf * 100:.1f}% confidence** to the class **{best}**." + chr(34) + ",\n"
    "            f" + chr(34) + "The next most likely class is {runner_up[0]} at {runner_up[1] * 100:.1f}% "
    "(separation: {spread * 100:.1f} points)." + chr(34) + ",\n"
    "        ]\n"
    "        if uncertain:\n"
    "            summary.append(" + chr(34) + "Confidence is below the reporting threshold - these findings are "
    "**inconclusive**, and a radiologist review is recommended." + chr(34) + ")\n"
    "        else:\n"
    "            summary.append(" + chr(34) + "Confidence is above the reporting threshold - the model considers this "
    "classification reliable for research purposes." + chr(34) + ")\n"
    "        summary.append(" + chr(34) + "These statements describe the model statistical outputs only. They are not "
    "clinical observations and must not be used for diagnosis." + chr(34) + ")\n"
    "        for s in summary:\n"
    "            st.markdown(" + chr(34) + "- " + chr(34) + " + s)"
)
assert old3 in src, "raw output anchor missing"
src = src.replace(old3, new3, 1)

roadmap = (
    "\n\nwith st.expander(" + chr(34) + "What is next - the roadmap to the next level" + chr(34) + "):\n"
    "    st.markdown("
    + chr(34) + chr(34) + chr(34)
    + "**1. Fine-tune the encoder** - the frozen EfficientNetB0 base unfrozen on its top blocks (low learning rate) - expected +1-3 points on the weak classes.\n\n"
    + "**2. External validation with hospital data** - the hospital proposal is ready; partner samples become the locked external test cohort - the industry-standard evidence of generalization.\n\n"
    + "**3. Language layer v2 - a grounded LLM narrator** - the model calibrated outputs and lung mask fed to a language model with strict grounding (it describes only what the model computed - no invented findings).\n\n"
    + "**4. Data ladder** - MIMIC-CXR (377k images, report-mined labels) and CheXpert (224k) - the corpora the field trains on; the pipeline absorbs them like the current seven datasets.\n\n"
    + "**5. DICOM ingestion + triage framing** - hospital-workflow integration: results flagged for radiologist review, never autonomous diagnosis.\n\n"
    + "**6. Drift monitoring** - per-scanner performance tracking over time - the layer where deployed models silently fail without it.\n\n"
    + "Research prototype only - not a medical diagnosis.\n"
    + chr(34) * 3 + ")"
)
src = src + roadmap
ast.parse(src)
p.write_text(src, encoding="utf-8")
print("app.py: abstention + findings summary + roadmap added")
