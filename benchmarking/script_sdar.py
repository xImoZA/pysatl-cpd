import ast
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from benchmarking.worker import Worker
from pysatl_cpd.core.algorithms.sdar_online_algorithm import SDARAlgorithm
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider


def construct_algorithm(order, forgetting_factor, smoothing_window_size, threshold):
    """Creates an instance of the SDAR algorithm."""
    return SDARAlgorithm(
        order=order,
        forgetting_factor=forgetting_factor,
        smoothing_window_size=smoothing_window_size,
        threshold=threshold,
    )


def benchmark_single_run(configuration, experiment_num, parameters, input_path):
    """
    Executes a single experiment run for a given configuration and parameter set.
    """
    try:
        order, forgetting_factor, smoothing_window_size, threshold = parameters

        # Define path and load data for the experiment
        data_path = Path(f"{input_path}/experiment/stage_1/{configuration}/sample_{experiment_num}/{configuration}")
        data = pd.read_csv(data_path / "sample.csv").to_numpy()
        data_provider = ListUnivariateProvider(data=list(data))

        algorithm = construct_algorithm(
            order=order,
            forgetting_factor=forgetting_factor,
            smoothing_window_size=smoothing_window_size,
            threshold=threshold,
        )

        cpd_core = OnlineCpdCore(algorithm=algorithm, data_provider=data_provider)
        worker = Worker(online_cpd=cpd_core)

        cp_results, cpd_time = worker.run()
        detected_cps = [result.change_point for result in cp_results]

        true_change_point = 250 if "-" in configuration else None

        return {
            "configuration": configuration,
            "experiment_number": experiment_num,
            "order": order,
            "forgetting_factor": forgetting_factor,
            "smoothing_window_size": smoothing_window_size,
            "threshold": threshold,
            "change_point": true_change_point,
            "detected_change_points": detected_cps,
            "work_time": cpd_time,
        }

    except Exception as e:
        print(f"Warning: Exp {experiment_num} for {configuration} with params {parameters} failed: {e}")
        return None


def run_benchmark(parameters_grid, configurations, num_of_experiments, input_path, save_path):
    """
    Runs the full benchmark for all specified parameter sets and configurations.
    Skips parameter sets that are already fully benchmarked.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_df = pd.read_csv(save_path) if save_path.exists() else pd.DataFrame()

    for params in parameters_grid:
        print(f"\n{'=' * 20}\nRunning benchmark for parameters: {params}\n{'=' * 20}")

        # Check if this parameter set has already been fully benchmarked
        if not history_df.empty:
            is_completed = (
                history_df.loc[
                    (history_df["order"] == params[0])
                    & (history_df["forgetting_factor"] == params[1])
                    & (history_df["smoothing_window_size"] == params[2])
                    & (history_df["threshold"] == params[3])
                ].shape[0]
                >= len(configurations) * num_of_experiments
            )
            if is_completed:
                print("This parameter set has already been benchmarked. Skipping.")
                continue

        # Run all experiments for all configurations in parallel using all available cores
        results = Parallel(n_jobs=-1)(
            delayed(benchmark_single_run)(config, exp_num, params, input_path)
            for config in configurations
            for exp_num in range(num_of_experiments)
        )

        # Collect valid results and append them to the CSV file
        valid_results = [res for res in results if res is not None]
        if valid_results:
            pd.DataFrame(valid_results).to_csv(save_path, mode="a", header=not save_path.exists(), index=False)
            print(f"Saved {len(valid_results)} results to {save_path}")


def evaluate_row(row, margin=25):
    """
    Calculates TP, FP, and FN for a single experiment row.
    """
    true_cp = row["change_point"]
    detected_cps = row["detected_change_points"]
    tp, fp, fn = 0, 0, 0

    if pd.notnull(true_cp):
        true_cp = int(true_cp)
        found_in_window = any(abs(detected - true_cp) <= margin for detected in detected_cps)

        if found_in_window:
            tp = 1
            if any(abs(detected - true_cp) > margin for detected in detected_cps):
                fp = 1
        else:
            fn = 1
            if detected_cps:
                fp = 1
    elif detected_cps:
        fp = 1

    return pd.Series({"TP": tp, "FP": fp, "FN": fn})


def evaluate_results(input_file, output_file, num_of_experiments):
    """
    Reads raw experiment results, calculates performance metrics, and saves them.
    """
    if not input_file.exists():
        print(f"Error: Results file not found at {input_file}")
        return

    df = pd.read_csv(input_file)
    df["detected_change_points"] = df["detected_change_points"].apply(ast.literal_eval)

    df.drop_duplicates(
        subset=[
            "configuration",
            "experiment_number",
            "order",
            "forgetting_factor",
            "smoothing_window_size",
            "threshold",
        ],
        inplace=True,
    )

    # Calculate metrics for each row
    metrics = df.apply(evaluate_row, axis=1)
    result = pd.concat([df, metrics], axis=1)

    # Group by parameters and aggregate metrics
    grouping_cols = ["configuration", "order", "forgetting_factor", "smoothing_window_size", "threshold"]
    final_metrics = (
        result.groupby(grouping_cols)
        .agg(
            TP=("TP", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum"),
            work_time_mean=("work_time", "mean"),
            work_time_std=("work_time", "std"),
        )
        .reset_index()
    )

    final_metrics["power"] = np.float64(final_metrics["TP"] / num_of_experiments)
    final_metrics["significance"] = np.float64(final_metrics["FP"] / num_of_experiments)

    to_round = ["power", "significance"]
    final_metrics[to_round] = final_metrics[to_round].round(4)

    final_metrics.to_csv(output_file, index=False)
    print(f"Evaluation complete. Final metrics saved to {output_file}")


def main():
    """Main function to run the benchmark and evaluation."""
    BASE_PATH = Path("/home/user/pysatl-cpd/")
    INPUT_DATA_PATH = BASE_PATH
    RESULTS_PATH = BASE_PATH / "results"

    RAW_RESULTS_FILE = RESULTS_PATH / "sdar_experiment_results.csv"
    FINAL_METRICS_FILE = RESULTS_PATH / "sdar_final_metrics.csv"

    # Define the parameter sets to be tested
    PARAMETERS_GRID = [
        (1, 0.97, 10, 1.556640625),
        (1, 0.97, 15, 1.037109375),
        (1, 0.99, 10, 3.96875),
        (1, 0.99, 15, 1.890625),
    ]

    # Load configuration names from the experiment description file
    CONFIGURATIONS = pd.read_csv(INPUT_DATA_PATH / "experiment/stage_1/experiment_description")["name"].to_list()
    NUM_OF_EXPERIMENTS = 1000

    run_benchmark(PARAMETERS_GRID, CONFIGURATIONS, NUM_OF_EXPERIMENTS, INPUT_DATA_PATH, RAW_RESULTS_FILE)
    evaluate_results(RAW_RESULTS_FILE, FINAL_METRICS_FILE, NUM_OF_EXPERIMENTS)


if __name__ == "__main__":
    main()
