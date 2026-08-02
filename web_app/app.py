#!/usr/bin/env python3
"""
STACPRED Web Application & REST API Server
Provides real-time binding free energy inference with dynamic reference structure selection.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import traceback
import joblib
from pathlib import Path
from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS

# Setup Relative Paths dynamically from repository root
WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
REF_DIR = REPO_ROOT / "data" / "references"
DEFAULT_REF = REF_DIR / "4tq_reference.mol2"
TEMPLATES_DIR = WEB_DIR / "templates"

# Ensure src/ is on system path to import prediction_core
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import prediction engine module
try:
    import inference.predict as prediction_core
    print("[SUCCESS] Loaded STACPRED inference module successfully!")
except ImportError as e:
    prediction_core = None
    print("[WARNING] Could not import 'inference.predict': " + str(e))

# Global Model Containers
BASE_MODELS = {}
STACKED_MODEL = None


def load_artifacts():
    """Pre-load models into memory on app launch."""
    global BASE_MODELS, STACKED_MODEL
    print("\n--- Loading STACPRED Ensemble Estimators into Memory ---")

    # 1. Load 4 Base XGBoost Models
    xgb_dir = MODELS_DIR / "base_models" / "xgb"
    if xgb_dir.exists():
        for pkl in xgb_dir.glob("*.pkl"):
            key = pkl.stem.replace("_final_results_2_model", "").upper() + "_XGB"
            BASE_MODELS[key] = joblib.load(pkl)
            print("  [XGBoost] Loaded " + str(key) + " from " + str(pkl.name))

    # 2. Load 4 Base MLP Models
    mlp_dir = MODELS_DIR / "base_models" / "mlp"
    if mlp_dir.exists():
        for pkl in mlp_dir.glob("*.pkl"):
            key = pkl.stem.replace("_final_results_2_mlp_model_tuned", "").upper() + "_MLP"
            BASE_MODELS[key] = joblib.load(pkl)
            print("  [MLP]     Loaded " + str(key) + " from " + str(pkl.name))

    # 3. Load Meta-Learner Stacking Model
    stacked_path = MODELS_DIR / "meta_learner" / "final_consensus_8model_stacking_model.pkl"
    if stacked_path.exists():
        STACKED_MODEL = joblib.load(stacked_path)
        print("  [RidgeCV] Loaded Stacking Meta-Learner from " + str(stacked_path.name))

    print("Total Base Models Loaded: " + str(len(BASE_MODELS)) + " / 8")
    print("-------------------------------------------------------\n")


def get_available_references():
    """Scans data/references/ for all available .mol2 reference files."""
    if not REF_DIR.exists():
        return {"4tq_reference": str(DEFAULT_REF)}
    
    refs = {}
    for f in REF_DIR.glob("*.mol2"):
        refs[f.stem] = str(f)
    return refs if refs else {"4tq_reference": str(DEFAULT_REF)}


# Execute pre-loading
load_artifacts()

# Flask App Configuration
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


# ==============================================================================
# REFERENCE MANAGEMENT API ENDPOINT
# ==============================================================================

@app.route("/get_references", methods=["GET"])
def get_references():
    """Returns available reference molecules to populate frontend dropdowns."""
    refs = get_available_references()
    return jsonify({
        "status": "success",
        "references": list(refs.keys()),
        "default": "4tq_reference" if "4tq_reference" in refs else list(refs.keys())[0]
    }), 200


# ==============================================================================
# DYNAMIC PAGE ROUTER
# ==============================================================================

@app.route("/")
@app.route("/index")
@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/<page>")
def render_page(page):
    if page == "predict" or page == "get_references":
        abort(404)

    clean_name = page.replace(".html", "")
    target_template = f"{clean_name}.html"

    if (TEMPLATES_DIR / target_template).exists():
        return render_template(target_template)

    return render_template("index.html")


# ==============================================================================
# REST API INFERENCE ENDPOINT
# ==============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not prediction_core:
            return jsonify({"error": "Inference module (prediction_core) is not available."}), 500

        if "mol2_file" not in request.files:
            return jsonify({"error": "No .mol2 file uploaded in request."}), 400

        file = request.files["mol2_file"]
        filename = file.filename

        if not filename:
            return jsonify({"error": "Invalid or empty filename."}), 400

        # Retrieve user-selected reference structure
        ref_choice = request.form.get("ref_choice", "4tq_reference")
        ref_dict = get_available_references()
        selected_ref_path = ref_dict.get(ref_choice, str(DEFAULT_REF))

        # Temporarily store incoming molecule
        temp_dir = WEB_DIR / "temp_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        saved_file_path = temp_dir / filename
        file.save(saved_file_path)

        print(f"\n[INFERENCE] Molecule: {filename} | Reference Template: {ref_choice}.mol2")

        # Run Consensus Inference
        results = prediction_core.predict_consensus(
            str(saved_file_path),
            selected_ref_path,
            BASE_MODELS,
            STACKED_MODEL
        )

        # Cleanup temporary uploaded file
        if saved_file_path.exists():
            saved_file_path.unlink()

        base_scores = {}
        final_score = 0.0
        profile_dict = {}

        if isinstance(results, tuple):
            if len(results) == 3:
                base_scores, final_score, profile_dict = results
            elif len(results) == 2:
                base_scores, final_score = results

        clean_base_scores = {}
        if isinstance(base_scores, dict):
            for k, v in base_scores.items():
                val = float(v[0]) if isinstance(v, (list, tuple)) else float(v)
                clean_base_scores[k] = val
                alias_key = k.replace("B2_", "") + "_PRED"
                clean_base_scores[alias_key] = val

        response = {
            "status": "success",
            "filename": filename,
            "reference_used": ref_choice,
            "base_scores": clean_base_scores,
            "final_score": float(final_score),
            "profile": profile_dict
        }

        print("[INFERENCE COMPLETE] Final Affinity Score: " + str(round(float(final_score), 4)) + " kcal/mol")
        return jsonify(response), 200

    except Exception as e:
        print("[ERROR] Exception caught during model execution:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Launching STACPRED Web Application at http://0.0.0.0:8000...")
    app.run(host="0.0.0.0", port=8000, debug=True)