import os
import pandas as pd
from datetime import datetime
from src.config import MODEL_ID, MODEL_NAME
from src.subspaces_analysis.subspace_benchmark import run_triple_experiment
from uncertainty_engine import UncertaintyAnalyzer
from probe import run_probing_experiment, analyze_probe_axes_orthogonality
from intervention import run_causal_intervention, plot_steering_results
from neuron_attribution import get_top_uncertainty_neurons
from plot import plot_layer_wise_emergence


def main():
    # --- PATH SETUP ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "src", "data", MODEL_NAME, "uncertainty_study_dataset.jsonl")
    baseline_path = os.path.join(project_root, "src", "data", MODEL_NAME, "baseline_dataset.jsonl")

    if not os.path.exists(data_path):
        print(f"❌ ERROR: Dataset not found at {data_path}")
        return

    # --- INITIALIZATION ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"{MODEL_NAME}_multi_layer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Set k dimension based on model architecture
    k_dim = -1
    if "gpt2" in MODEL_NAME:
        k_dim = 12
    elif "Llama" in MODEL_NAME:
        k_dim = 4
    elif "gemma" in MODEL_NAME or "Qwen" in MODEL_NAME:
        k_dim = 5
    else:
        raise Exception(f"Unknown model: {MODEL_NAME}")

    analyzer = UncertaintyAnalyzer(model_name=MODEL_ID, k=k_dim)

    # --- LAYER SELECTION ---
    total_layers = analyzer.model.cfg.n_layers
    step = 4
    layers_to_check = list(range(0, total_layers, step))
    if (total_layers - 1) not in layers_to_check:
        layers_to_check.append(total_layers - 1)

    print(f"🔍 Analyzing layers: {layers_to_check}")

    # --- STEP 1: GEOMETRIC ANALYSIS ---
    # summary_df contains metrics across all layers
    # all_layer_storage is a dict mapping layer_idx to its activation storage
    csv_metrics_path = os.path.join(output_dir, "multi_layer_metrics.csv")
    summary_df, all_layer_storage = run_triple_experiment(
        analyzer, data_path, baseline_path, output_dir, layers_to_test=layers_to_check
    )
    summary_df.to_csv(csv_metrics_path, index=False)
    plot_layer_wise_emergence(csv_metrics_path, output_dir)

    # Collector for probing results across all layers
    all_probing_results = []

    # --- NEW: PER-LAYER FUNCTIONAL ANALYSIS LOOP ---
    for layer_idx in layers_to_check:
        print(f"\n" + "=" * 50)
        print(f"🏗️  PROCESSING LAYER {layer_idx}")
        print("=" * 50)

        # Create layer-specific subdirectories to avoid overwriting reports
        layer_output_dir = os.path.join(output_dir, f"layer_{layer_idx}")
        os.makedirs(layer_output_dir, exist_ok=True)

        storage_current_layer = all_layer_storage[layer_idx]

        # --- STEP 2: PROBING ---
        print(f"🧠 Probing layer {layer_idx} for linear disentanglement...")
        # run_probing_experiment handles CV and fits models for the specific layer
        probing_results_df, probing_models_dict = run_probing_experiment(storage_current_layer)

        # Metadata tagging for final aggregation
        probing_results_df['Layer'] = layer_idx
        all_probing_results.append(probing_results_df)

        # Calculate geometric orthogonality for this specific layer
        analyze_probe_axes_orthogonality(probing_models_dict, layer_output_dir)

        # --- STEP 3: NEURON ATTRIBUTION ---
        # Map probe weights back to specific neurons for this layer
        print(f"🔍 Attributing uncertainty signals to neurons in layer {layer_idx}...")
        get_top_uncertainty_neurons(probing_models_dict, "Detection (Cert vs Uncert)", "Null Space", layer_output_dir)
        get_top_uncertainty_neurons(probing_models_dict, "Type (Epi vs Alea)", "Null Space", layer_output_dir)

    # Save aggregated probing results for all checked layers
    final_probing_df = pd.concat(all_probing_results, ignore_index=True)
    final_probing_df.to_csv(os.path.join(output_dir, "all_layers_probing_results.csv"), index=False)

    # --- STEP 4: CAUSAL INTERVENTION (Steering) ---
    # Typically performed using the final layer's regulatory vectors
    final_layer_idx = layers_to_check[-1]
    print(f"\n🧪 Running Causal Steering Intervention (Layer {final_layer_idx})...")

    test_data = [
        {"prompt": "The capital of France is", "logic": "Paris"},
        {"prompt": "Two plus two equals", "logic": "four"}
    ]
    test_prompts_df = pd.DataFrame(test_data)
    steering_df = run_causal_intervention(
        analyzer,
        all_layer_storage[final_layer_idx],
        test_prompts_df
    )

    steering_path = os.path.join(output_dir, "steering_results.csv")
    steering_df.to_csv(steering_path, index=False)
    plot_steering_results(steering_path)

    print(f"\n✨ DONE. All results saved in: {output_dir}")


if __name__ == "__main__":
    main()