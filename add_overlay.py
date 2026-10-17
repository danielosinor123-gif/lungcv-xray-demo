import ast
from pathlib import Path

p = Path("app.py")
src = p.read_text(encoding="utf-8-sig")

old1 = (
    "        raw = model.predict(x, verbose=0)\n"
    "        probs = raw[0] if isinstance(raw, list) else raw\n"
    "        if getattr(probs, 'ndim', 1) == 2:\n"
    "            probs = probs[0]"
)
new1 = (
    "        raw = model.predict(x, verbose=0)\n"
    "        probs = raw[0] if isinstance(raw, list) else raw\n"
    "        if getattr(probs, 'ndim', 1) == 2:\n"
    "            probs = probs[0]\n"
    "        seg_mask = None\n"
    "        if isinstance(raw, list) and len(raw) > 1:\n"
    "            seg_mask = np.asarray(raw[1][0, :, :, 0], dtype=np.float32)"
)
assert old1 in src, "predict anchor missing"
src = src.replace(old1, new1, 1)

old2 = "        with st.expander(" + chr(34) + "Raw output (JSON)" + chr(34) + "):"
new2 = (
    "        st.subheader(" + chr(34) + "Lung region (model attention)" + chr(34) + ")\n"
    "        if seg_mask is not None:\n"
    "            base_img = np.asarray(Image.open(image_source).convert(" + chr(34) + "RGB" + chr(34) + ").resize((224, 224), Image.LANCZOS), dtype=np.float32)\n"
    "            m = seg_mask > 0.5\n"
    "            overlay = base_img.copy()\n"
    "            overlay[m] = overlay[m] * 0.45 + np.array([0.0, 200.0, 0.0]) * 0.55\n"
    "            st.image(overlay.astype(np.uint8), caption=" + chr(34) + "Green: lung regions the model traced (model attention, not clinical segmentation)" + chr(34) + ", use_container_width=True)\n"
    "        else:\n"
    "            st.caption(" + chr(34) + "This model has no segmentation head - lung-region overlay unavailable." + chr(34) + ")\n"
    "\n"
    "        with st.expander(" + chr(34) + "Raw output (JSON)" + chr(34) + "):"
)
assert old2 in src, "expander anchor missing"
src = src.replace(old2, new2, 1)

ast.parse(src)
p.write_text(src, encoding="utf-8")
print("app.py: mask capture + overlay panel added")
