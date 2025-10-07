from pathlib import Path

import pandas as pd

from benchmarking.worker import Worker
from pysatl_cpd.core.algorithms.sdar_online_algorithm import SDARAlgorithm
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider

MIN_DIFF = 0.01


def construct_algorithm(order, forgetting_factor, smoothing_window_size, threshold):
    """Creates an instance of the SDAR algorithm."""
    return SDARAlgorithm(
        order=order,
        forgetting_factor=forgetting_factor,
        smoothing_window_size=smoothing_window_size,
        threshold=threshold,
    )


def benchmark_for_threshold(configuration, num_of_experiments, parameters, input_path):
    """
    Runs a series of experiments for a given set of parameters (including the threshold)
    and returns the results as a list.
    """
    order, forgetting_factor, smoothing_window_size, threshold = parameters

    buffer = []
    for experiment in range(num_of_experiments):
        data_path = Path(f"{input_path}/experiment/stage_0/{configuration}/sample_{experiment}/{configuration}")
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

        detected_change_points = [result.change_point for result in cp_results]
        buffer.append(detected_change_points)

    return buffer


def get_significance_for_threshold(
    parameters_to_test: tuple,
    test_threshold: float,
    configuration: str,
    num_of_experiments: int,
    input_path: Path,
):
    """
    Calculates the significance level for threshold.
    """
    order, forgetting_factor, smooth_window_size = parameters_to_test
    parameters_with_threshold = (order, forgetting_factor, smooth_window_size, test_threshold)

    print(f"\n--- Testing threshold: {test_threshold:.4f} for configuration: '{configuration}' ---")
    detected_cps_list = benchmark_for_threshold(
        configuration=configuration,
        num_of_experiments=num_of_experiments,
        parameters=parameters_with_threshold,
        input_path=input_path,
    )

    # Count False Positives
    fp_sum = sum(1 for detected_points in detected_cps_list if len(detected_points) > 0)
    significance = fp_sum / num_of_experiments

    print(f"--- Result for threshold {test_threshold:.4f}: FP={fp_sum}, Significance Level={significance:.4f} ---")
    return significance


def find_optimal_threshold(parameters_to_test: tuple, target_significance: float, configuration: str, input_path: Path):
    """
    Finds the optimal threshold using a binary search to achieve the
    target significance level.
    """
    order, ff, sws = parameters_to_test
    print(
        f"\n{'=' * 35}\n"
        f"Starting search for parameters: o={order}, ff={ff}, sws={sws}\n"
        f"Target significance level = {target_significance} for '{configuration}'\n"
        f"{'=' * 35}"
    )

    # Binary search settings
    tolerance = target_significance * 0.1
    max_iterations = 20
    low_threshold = 1.0
    high_threshold = 20.0
    num_of_experiments = 2500

    final_significance = -1.0

    for i in range(max_iterations):
        mid_threshold = (low_threshold + high_threshold) / 2

        # Stopping condition if the range becomes too narrow
        if (high_threshold - low_threshold) < MIN_DIFF:
            print("Search range is too small, stopping.")
            break

        final_significance = get_significance_for_threshold(
            parameters_to_test=parameters_to_test,
            test_threshold=mid_threshold,
            configuration=configuration,
            num_of_experiments=num_of_experiments,
            input_path=input_path,
        )

        if abs(final_significance - target_significance) <= tolerance:
            print(f"\nSuccess! Universal threshold found for target {target_significance}.")
            break

        if final_significance > target_significance:
            low_threshold = mid_threshold
        else:
            high_threshold = mid_threshold

        print(f"Iteration {i + 1}: New search range [{low_threshold:.4f}, {high_threshold:.4f}]")

    # Calculate the final threshold
    final_threshold = (low_threshold + high_threshold) / 2
    if i == max_iterations - 1:
        print(f"\nMax iterations reached for target {target_significance}.")

    return {
        "configuration": configuration,
        "order": order,
        "forgetting_factor": ff,
        "smoothing_window_size": sws,
        "target_significance": target_significance,
        "optimal_threshold": final_threshold,
        "final_significance": final_significance,
    }


if __name__ == "__main__":
    INPUT_PATH = Path("input/path")
    OUTPUT_PATH = Path("output/path")
    SAVE_FILENAME = "sdar_results/optimal_thresholds_sdar.csv"

    CONFIGURATION = "normal"
    TARGET_SIGNIFICANCE = 0.05
    # List of parameter sets to test
    PARAM_GRID = [(1, 0.99, 10), (1, 0.99, 15), (1, 0.97, 10), (1, 0.97, 15)]

    all_results = []
    for params in PARAM_GRID:
        try:
            result = find_optimal_threshold(
                parameters_to_test=params,
                target_significance=TARGET_SIGNIFICANCE,
                configuration=CONFIGURATION,
                input_path=INPUT_PATH,
            )
            if result:
                all_results.append(result)
        except Exception as exc:
            print(f"Task for parameters {params} generated an exception: {exc}")

    print(f"\n\n{'=' * 30}\nAll searches for '{CONFIGURATION}' are complete!\n{'=' * 30}")

    if all_results:
        final_df = (
            pd.DataFrame(all_results)
            .sort_values(by=["target_significance", "order", "forgetting_factor"])
            .reset_index(drop=True)
        )

        print("Final results:")
        print(final_df.to_string())

        save_file = OUTPUT_PATH / SAVE_FILENAME
        final_df.to_csv(save_file, index=False)
    else:
        print("No results were found.")
