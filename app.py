"""Chest X-Ray Classification - Streamlit front end for FarhanKO/lung-disease-prediction.

Research and education only. Not a medical device.
"""

from __future__ import annotations

import hashlib

import altair as alt
import pandas as pd
import streamlit as st

import model_utils as mu  # referenced as a module so tests can swap in a mock loader

st.set_page_config(
    page_title="Chest X-Ray Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#1565C0"
LIGHT_BLUE = "#9DBFE8"

DISCLAIMER = (
    "**Research and education only.** This application is not a medical device, has not been "
    "clinically validated, and must not be used for diagnosis or treatment decisions. "
    "Always consult a qualified healthcare professional."
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
      .app-header {{
        background: linear-gradient(135deg, {BLUE} 0%, #1E88E5 100%);
        color: #FFFFFF; border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
      }}
      .app-header h1 {{ color: #FFFFFF; margin: 0; padding: 0; font-size: clamp(1.5rem, 4vw, 2.3rem); }}
      .app-header p {{ margin: .4rem 0 0 0; font-size: clamp(.9rem, 2.2vw, 1.05rem); opacity: .95; }}
      @media (max-width: 640px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 1rem; }}
        .app-header {{ padding: 1rem 1.1rem; }}
      }}
    </style>
    <div class="app-header">
      <h1>Chest X-Ray Classification</h1>
      <p>A research demo that runs a pretrained deep-learning model (DenseNet121) to suggest one of
      six chest X-ray classes, and flags images it is unsure about or cannot classify.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- model cache


@st.cache_resource(
    show_spinner="Loading the model. The first start downloads about 40 MB and can take a minute or two..."
)
def get_classifier():
    """Download and initialise the classifier once per server process (not on every rerun).

    Exceptions are not cached by Streamlit, so a failed download can simply be retried.
    """
    return mu.load_classifier()


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("About")
    st.write(
        "Upload a chest X-ray and the model returns the most likely class among "
        "**Covid-19, Emphysema, Normal, Pneumonia-Bacterial, Pneumonia-Viral** and **Tuberculosis**."
    )
    st.subheader("How it works")
    st.markdown(
        "1. An out-of-distribution check rejects images that do not look like chest X-rays.\n"
        "2. A fine-tuned DenseNet121 scores the six classes.\n"
        f"3. If the top score is below {mu.CONFIDENCE_THRESHOLD:.0%}, the result is flagged for review."
    )
    st.subheader("Model attribution")
    st.markdown(
        f"Model: [`{mu.REPO_ID}`]({mu.REPO_URL}) by FarhanKO on Hugging Face "
        "(license listed as *other*; see the model card). Weights are downloaded from the Hub at runtime."
    )
    st.subheader("Disclaimer")
    st.warning(DISCLAIMER)
    st.caption(
        "Scores are raw model outputs and are not calibrated medical probabilities. "
        "Uploaded images are processed in memory and are not stored by this app."
    )

st.info(DISCLAIMER, icon="⚠️")

# --------------------------------------------------------------------------- result rendering


def render_result(result: mu.ClassificationResult) -> None:
    info = mu.status_info(result.status)
    banner = {"success": st.success, "warning": st.warning, "error": st.error}[info.severity]
    banner(f"**Status: {info.title}.** {info.explanation}")
    if result.message:
        st.caption(result.message)

    if result.is_rejected:
        st.markdown("#### No classification")
        st.write("The system could not classify this image, so no class is shown.")
        return

    left, right = st.columns(2)
    with left:
        if result.status == "accepted":
            st.metric("Predicted class", result.label)
        else:
            st.metric("Best guess (unconfirmed)", result.label)
    with right:
        st.metric("Model confidence", mu.format_percent(result.confidence))
    st.progress(min(max(result.confidence, 0.0), 1.0))
    st.caption(
        "Confidence is the model's raw top-class softmax score. It is not a calibrated or "
        "clinically validated probability."
    )

    if result.p_abnormal is not None:
        st.caption(
            f"Abnormality score (1 - P(Normal)): {mu.format_percent(result.p_abnormal)}. "
            "A raw model score, not a diagnosis."
        )

    if result.top_3:
        st.markdown("#### Top 3 predictions")
        top_df = pd.DataFrame(
            [
                {"Rank": i, "Class": name, "Score": mu.format_percent(score)}
                for i, (name, score) in enumerate(result.top_3, start=1)
            ]
        )
        st.table(top_df.set_index("Rank"))

    rows = mu.probability_rows(result)
    if rows:
        st.markdown("#### Probability distribution")
        chart_df = pd.DataFrame(rows, columns=["Class", "Probability"])
        top_class = rows[0][0]
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Probability:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", title="Model score"),
                ),
                y=alt.Y("Class:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.Class == top_class, alt.value(BLUE), alt.value(LIGHT_BLUE)
                ),
                tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")


# --------------------------------------------------------------------------- main flow

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="JPG, JPEG or PNG, up to 10 MB.",
)

image = None
file_id = None
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    file_id = hashlib.sha256(raw_bytes).hexdigest()
    try:
        image = mu.load_image(raw_bytes)
    except mu.InvalidImageError as exc:
        st.error(f"Invalid image: {exc}")

preview_col, result_col = st.columns(2, gap="large")

with preview_col:
    st.subheader("Image preview")
    if image is not None:
        st.image(image, caption=f"{uploaded.name} ({image.width} x {image.height}px)", width="stretch")
    elif uploaded is None:
        st.info("No image yet. Upload a JPG, JPEG or PNG chest X-ray above to get started.")

with result_col:
    st.subheader("Result")
    clicked = st.button("Classify X-ray", type="primary", width="stretch")

    if clicked:
        if uploaded is None:
            st.warning("Please upload a chest X-ray image first.")
        elif image is not None:
            try:
                classifier = get_classifier()
                with st.spinner("Analysing the image..."):
                    result = mu.classify(classifier, image)
                st.session_state["outcome"] = {"file_id": file_id, "result": result}
            except mu.ClassifierError as exc:
                st.session_state["outcome"] = {
                    "file_id": file_id,
                    "error": str(exc),
                    "kind": type(exc).__name__,
                }

    outcome = st.session_state.get("outcome")
    if outcome and outcome["file_id"] == file_id and file_id is not None:
        if "error" in outcome:
            if outcome["kind"] == "ModelLoadError":
                st.error(f"Model unavailable. {outcome['error']}")
                st.caption("Press **Classify X-ray** again to retry the download.")
            elif outcome["kind"] == "InferenceError":
                st.error(f"Classification failed. {outcome['error']}")
            else:
                st.error(f"Unexpected model output. {outcome['error']}")
        else:
            render_result(outcome["result"])
    elif image is not None and not clicked:
        st.info("Press **Classify X-ray** to run the model on this image.")
    elif uploaded is None and not clicked:
        st.caption("Results will appear here.")

# --------------------------------------------------------------------------- footer

st.divider()
st.caption(
    f"Model: [{mu.REPO_ID}]({mu.REPO_URL}) · Research and education only · "
    "Not a medical device · Not clinically validated."
)
