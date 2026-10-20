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
    initial_sidebar_state="collapsed",
)

BLUE = "#1565C0"
BLUE_DARK = "#0D47A1"
LIGHT_BLUE = "#9DBFE8"

DISCLAIMER = (
    "**Research and education only.** This application is not a medical device, has not been "
    "clinically validated, and must not be used for diagnosis or treatment decisions."
)

# The styling below does not depend on .streamlit/config.toml, so the look is the same even if
# that file is missing from the repository.
st.markdown(
    f"""
    <style>
      .stApp {{ background: #F4F8FD; }}
      [data-testid="stHeader"] {{ background: transparent; }}
      .block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1100px; }}

      .hero {{
        background: linear-gradient(120deg, {BLUE_DARK} 0%, {BLUE} 55%, #2B8CE6 100%);
        border-radius: 18px; padding: 1.6rem 1.9rem; margin-bottom: 1rem; color: #fff;
        box-shadow: 0 6px 18px rgba(21, 101, 192, .22);
      }}
      .hero h1 {{
        color: #fff; margin: 0; padding: 0; font-weight: 700; line-height: 1.15;
        font-size: clamp(1.55rem, 4.2vw, 2.3rem);
      }}

      [data-testid="stVerticalBlockBorderWrapper"] {{
        background: #FFFFFF; border-radius: 16px; border: 1px solid #DCE7F5;
        box-shadow: 0 2px 10px rgba(21, 101, 192, .06);
      }}
      [data-testid="stFileUploader"] section {{
        border: 2px dashed {LIGHT_BLUE}; background: #F5F9FF; border-radius: 12px;
      }}
      button[kind="primary"], [data-testid="stBaseButton-primary"] {{
        background: {BLUE}; border: 1px solid {BLUE}; color: #fff; border-radius: 10px;
        font-weight: 600; padding: .55rem 1rem;
      }}
      button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
        background: {BLUE_DARK}; border-color: {BLUE_DARK}; color: #fff;
      }}
      [data-testid="stMetricValue"] {{ color: {BLUE_DARK}; }}

      @media (max-width: 640px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 3.4rem; }}
        .hero {{ padding: 1.1rem 1.2rem; border-radius: 14px; }}
      }}
    </style>

    <div class="hero">
      <h1>Chest X-Ray Classification</h1>
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
        "A research demo that runs a pretrained chest X-ray model. It is meant for learning and "
        "experimentation, not for clinical use."
    )
    st.subheader("How it works")
    st.markdown(
        "1. A check rejects images that do not look like chest X-rays.\n"
        "2. A fine-tuned DenseNet121 scores the image.\n"
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
            .mark_bar(cornerRadiusEnd=4)
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

upload_col, result_col = st.columns(2, gap="large")

uploaded = None
image = None
file_id = None

with upload_col:
    with st.container(border=True):
        st.subheader("1. Upload image")
        uploaded = st.file_uploader(
            "Upload a chest X-ray image",
            type=["jpg", "jpeg", "png"],
            max_upload_size=mu.MAX_UPLOAD_BYTES // (1024 * 1024),
            label_visibility="collapsed",
            help="JPG, JPEG or PNG, up to 10 MB.",
        )
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            file_id = hashlib.sha256(raw_bytes).hexdigest()
            try:
                image = mu.load_image(raw_bytes)
            except mu.InvalidImageError as exc:
                st.error(f"Invalid image: {exc}")

        if image is not None:
            st.image(image, caption=f"{uploaded.name} ({image.width} x {image.height}px)", width="stretch")
        elif uploaded is None:
            st.info("No image yet. Upload a JPG, JPEG or PNG chest X-ray to get started.")

with result_col:
    with st.container(border=True):
        st.subheader("2. Result")
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
            st.caption("Your result will appear here after you upload an image and press the button.")
