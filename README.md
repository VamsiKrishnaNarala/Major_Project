# Chest X-Ray Classification (Streamlit)

A small web app where you upload a chest X-ray (JPG, JPEG or PNG) and a **pretrained** deep-learning model suggests one of six classes. It is built for free hosting on **Streamlit Community Cloud** and downloads the model from Hugging Face when it starts, so no model files live in this repository.

> **Research and education only.** This is not a medical device. It has not been clinically validated and must not be used for diagnosis or treatment decisions.

---

## 1. Project overview

- Front end: Streamlit (blue/white theme, responsive layout, sidebar with attribution and disclaimer).
- Model: the existing `FarhanKO/lung-disease-prediction` DenseNet121 checkpoint. **No training is done here and no other model is substituted.**
- Inference: the model author's own `inference.py` (`ChestXrayClassifier`), downloaded at runtime.
- Runs on CPU. No API keys, tokens or paid services.

## 2. Supported classes

In the exact order used by the model:

1. Covid-19
2. Emphysema
3. Normal
4. Pneumonia-Bacterial
5. Pneumonia-Viral
6. Tuberculosis

## 3. Model attribution

- Model: **[FarhanKO/lung-disease-prediction](https://huggingface.co/FarhanKO/lung-disease-prediction)** on Hugging Face, by FarhanKO.
- Checkpoint used: `DenseNet121_FT_best.keras` (about 36 MB), plus `metadata.json`, `lung_ood_detector.joblib` and `inference.py`.
- All credit for the model and inference code belongs to its author. This app is only a front end.
- What the model card reports (self-reported, on its own test split): accuracy about 0.847 and macro-F1 about 0.846. The card also lists limitations, for example that the data was split by image rather than by patient (so scores may be optimistic), that bacterial vs viral pneumonia is hard to separate (F1 about 0.70-0.74), and that lateral views, paediatric X-rays and phone photos of films are untested. Read the [model card](https://huggingface.co/FarhanKO/lung-disease-prediction) before drawing any conclusion.

## 4. Architecture and inference pipeline

```text
Browser ── upload ──> app.py (Streamlit UI)
                         │  validates + decodes the file  (model_utils.load_image)
                         ▼
                   model_utils.py
                         │  first click only: snapshot_download() of 4 files from Hugging Face,
                         │  then imports the official inference.py and builds ChestXrayClassifier
                         │  (cached with st.cache_resource, so this happens once per server process)
                         ▼
        ChestXrayClassifier.predict(PIL image)          <- official code
          1. preprocess: RGB, resize 224x224 (PIL BILINEAR), float32, divide by 255
          2. OOD gate (Isolation Forest on DenseNet121 embeddings)  -> "rejected"
          3. DenseNet121_FT softmax over 6 classes
          4. top score < 0.75 -> "review", otherwise "accepted"
                         ▼
                   model_utils.normalize_result   (validates the returned dict)
                         ▼
                   app.py renders status, class, confidence, top-3 and probability chart
```

Facts confirmed by reading the real `inference.py`, `metadata.json` and `requirements.txt` (October 2026):

- `predict()` accepts a **file path, a PIL image or a uint8 NumPy array**. This app passes a validated RGB PIL image.
- `ChestXrayClassifier(source)` accepts either a Hugging Face repo id or a **local folder**. This app downloads just the four needed files with `snapshot_download(allow_patterns=...)` and passes the local folder, so the other six checkpoints (about 650 MB) are never fetched.
- The class loads only `DenseNet121_FT` unless you ask for the ensemble (the app does not).
- The class sets `KERAS_BACKEND=torch` by default when PyTorch is installed; the app also sets it before importing Keras.

Files in this project:

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `model_utils.py` | Image validation, model loading, result validation (no Streamlit imports) |
| `tests/test_app_utils.py` | Unit tests and mock-classifier UI tests |
| `requirements.txt` | Pinned dependencies |
| `.streamlit/config.toml` | Theme, 10 MB upload limit, usage stats off |
| `.gitignore` | Keeps model files, caches and secrets out of Git |

No `packages.txt` is needed: no system packages are required.

## 5. Python version and dependencies

Use **Python 3.13** (the version the model author tested). Pinned versions in `requirements.txt`:

| Package | Version | Why |
| --- | --- | --- |
| keras | 3.14.1 | tested by the model author |
| torch | 2.11.0 (CPU build) | tested by the model author; Keras backend |
| scikit-learn | 1.8.0 | **must be 1.8.x**: the OOD gate is a pickle made with this version |
| streamlit | 1.66.0 | version the UI tests were run with |
| numpy, pillow, joblib, huggingface_hub, pandas, altair | version ranges | no known conflicts |

Do not just "upgrade everything": the latest Keras, PyTorch, NumPy and scikit-learn are not guaranteed to work together, and a different scikit-learn minor version may fail to load the OOD gate.

The first line of `requirements.txt` points pip at PyTorch's CPU wheel index. The default Linux wheel on PyPI bundles several GB of CUDA libraries, which is far too large for free hosting.

## 6. Run locally on Windows

1. Install **Python 3.13** from [python.org](https://www.python.org/downloads/windows/). Tick **"Add python.exe to PATH"** in the installer.
2. Unzip `lung-disease-classifier-streamlit.zip` and open **PowerShell** in the extracted `lung-disease-classifier` folder.
3. Create and activate a virtual environment:
   ```powershell
   py -3.13 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again. (Or use Command Prompt and run `.venv\Scripts\activate.bat`.)
4. Install the dependencies (a few hundred MB, takes several minutes):
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   streamlit run app.py
   ```
   Your browser opens at `http://localhost:8501`. Upload an image and press **Classify X-ray**. The first classification downloads about 40 MB from Hugging Face and may take a minute or two.
6. Optional, run the tests:
   ```powershell
   python -m pip install pytest
   python -m pytest -v
   ```

## 7. Upload the project to GitHub

Streamlit Community Cloud deploys from a GitHub repository. `app.py` and `requirements.txt` must be at the **root** of the repository.

**Option A: website (no Git needed)**

1. Create a free account at [github.com](https://github.com) and click **New repository**. Name it, for example, `lung-disease-classifier`, choose **Public**, and create it.
2. Unzip the ZIP on your computer.
3. On the new repository page click **uploading an existing file**, then drag in **the contents** of the `lung-disease-classifier` folder (`app.py`, `model_utils.py`, `requirements.txt`, `README.md`, `.gitignore`, and the `tests` and `.streamlit` folders). Make sure `.streamlit/config.toml` is included: hidden-looking folders are easy to miss. After uploading, check that `.streamlit/config.toml` is listed in the repository.
4. Click **Commit changes**.

**Option B: Git command line**

```powershell
cd lung-disease-classifier
git init
git add .
git commit -m "Chest X-ray classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/lung-disease-classifier.git
git push -u origin main
```

Do not upload model files. `.gitignore` already blocks `*.keras` and `*.joblib`.

## 8. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub. Authorise access to your repository if asked.
2. Click **Create app** and choose **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/lung-disease-classifier`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings** and select **Python 3.13**. (Python 3.13 is the version the model was tested with. Other versions are untested here.)
5. Optional: in **Secrets**, add `MODEL_REVISION = "<commit hash>"` to pin the model files to one exact Hugging Face commit (see the trust note in section 11).
6. Click **Deploy**. The first build installs the dependencies and can take several minutes. The page itself loads quickly; the model is downloaded the first time someone presses **Classify X-ray**.
7. Open the app URL, upload a chest X-ray and press **Classify X-ray**. Expect the first result to take longer (model download and PyTorch/Keras start-up).

Free apps go to sleep after a period of inactivity. When someone opens a sleeping app it restarts and the model must be downloaded again (a "cold start").

## 9. Troubleshooting

**Dependency conflicts or the build fails**
- Make sure the app is set to Python 3.13 in Advanced settings (delete and redeploy the app if you need to change it).
- Do not loosen the pins for `keras`, `torch` or `scikit-learn`.
- If the build fails on the `--extra-index-url` line or on `torch==2.11.0`, try removing that first line. The default PyPI wheel works but is much larger and may exceed the free resource limits. Another option is to switch backend: replace `torch` with `tensorflow-cpu` in `requirements.txt` and add the Streamlit secret `KERAS_BACKEND = "tensorflow"`. The model author states TensorFlow also works, but this was not tested here.
- Errors or warnings about unpickling `lung_ood_detector.joblib` (for example `InconsistentVersionWarning`): scikit-learn is probably not 1.8.x. Reinstall with `scikit-learn==1.8.0`.

**Out of memory / app restarts or shows "resource limits"**
- Community Cloud gives limited RAM (the amount has changed over time; check the current Streamlit docs). Keras and PyTorch alone need a few hundred MB. This app loads a single model and runs one prediction at a time, but there is no extra headroom to spare.
- Reboot the app from the Streamlit dashboard, avoid uploading very large images (the limit is 10 MB and 40 megapixels), and do not enable the ensemble option.

**Slow first request (cold start)**
- Normal after the app wakes up: it installs nothing new, but it downloads about 40 MB and imports Keras and PyTorch. Later requests are fast because the model object is cached.

**"Model unavailable" / failed model download**
- Hugging Face may be temporarily unreachable or rate-limiting. Press **Classify X-ray** again; failed loads are not cached, so it retries.
- If you pinned `MODEL_REVISION`, check the hash is correct, or remove the setting to follow `main`.
- Behind a firewall, make sure `huggingface.co` is reachable.

**"The model was downloaded but could not be initialised"**
- Most often a dependency version mismatch (see section 5). The message includes the underlying error.

**Invalid image messages**
- Only JPEG and PNG files up to 10 MB, at least 32 px per side, are accepted. Re-export DICOM images as PNG or JPG first.

## 10. Status values explained

| Status | Meaning | What the app shows |
| --- | --- | --- |
| `accepted` | The top-class score is at least 0.75. | Predicted class, confidence, top-3, full distribution, with a reminder that it is not a diagnosis. |
| `review` | The top-class score is below 0.75. | A warning, and the class is labelled **"Best guess (unconfirmed)"**. On the model's own test set only about 63% of flagged images were correct. |
| `rejected` | The out-of-distribution gate says the image does not look like a chest X-ray. | An error-style notice and **no class, no confidence and no probabilities**. The app never invents a class for a rejected image. |

If the classifier ever reports `accepted` with a score below 0.75, the app treats it as `review` and says so.

**About "confidence":** the number shown is the model's raw top-class softmax output. It is **not** a calibrated or clinically validated probability. A score of 99% does not mean there is a 99% chance the person has the condition. The OOD gate was only measured against X-rays and noise/blank images; the model card says other image types (CT slices, ECG printouts, photos) have not been measured, so an unsuitable image can still be accepted.

## 11. Disclaimer, trust and licensing

- **Not a medical device and not clinically validated.** Do not use it to make or influence decisions about anyone's health. Do not rely on a "Normal" result to rule out disease.
- **Privacy:** uploaded images are decoded in memory by this app and are not saved by it. Still, do not upload images that identify a patient, and remember that a public Streamlit app runs on third-party infrastructure.
- **Trust assumption:** at runtime this app downloads `inference.py` and runs it, and loads `lung_ood_detector.joblib`, which is a Python pickle. Loading a pickle can execute code, so you are trusting the model author's repository and Hugging Face. By default the app follows the `main` branch. For a stable and safer setup, set the environment variable or Streamlit secret `MODEL_REVISION` to a specific commit hash that you have reviewed. When this project was prepared the Hub API reported commit `a2c8da8837c8195be7df87b170638c36a9051284` (last modified 2026-09-29); this was read through a fetch tool and not independently verified, so confirm it on the repository's commit history before pinning.
- **Licensing:** the model repository lists its license as **"other"** without naming terms. Check the model card and contact the author before any use beyond personal research or education, and especially before commercial use. The training data comes from public chest X-ray collections that have their own licences. This project's own code ships **without a LICENSE file**; add one that suits you before you publish it. Streamlit, Keras, PyTorch and the other libraries are under their own licences.

## 12. Testing status (honest summary)

Run in a sandbox with Python 3.13.16, streamlit 1.66.0, pytest 9.1.1, numpy 2.5.4, pillow 12.3.0, huggingface_hub 2.2.0, pandas 3.0.6, altair 6.3.0:

- Python syntax check of all files: passed.
- `python -m pytest`: **69 tests passed.** They cover image loading and validation (RGB conversion, grayscale, RGBA, 16-bit PNG, corrupt/empty/oversized/unsupported files), result validation and formatting for all three statuses, malformed model output, the model-loading glue (with the Hugging Face download and the model class replaced by test doubles), and the full Streamlit UI driven by Streamlit's `AppTest` with a **mock classifier** (upload, missing upload, invalid file, accepted/review/rejected display, load failure, inference failure, model loaded only once, result cleared on a new upload). The mock results are test doubles, not model predictions.

**Not tested:**

- **No real inference was run.** The sandbox could not reach Hugging Face, so the real model files were never downloaded, and no real X-ray was classified.
- keras 3.14.1, torch 2.11.0 and scikit-learn 1.8.0 were not installed there. Their versions come from the model repository's own `requirements.txt` ("tested with keras 3.14.1, torch 2.11, scikit-learn 1.8.0, Python 3.13"); that they install together on Streamlit Community Cloud, including the CPU-wheel index line, has not been verified.
- The real `snapshot_download` call and the real `ChestXrayClassifier` initialisation from a local folder were checked only by reading the model's source code, not by running them.
- The app was not deployed to Streamlit Community Cloud, and memory use there is unmeasured.
- Looks on real phones and browsers were not checked visually; the layout relies on Streamlit's built-in column stacking plus a small CSS media query.

Please deploy it and do a first classification yourself to confirm the end-to-end path works.
