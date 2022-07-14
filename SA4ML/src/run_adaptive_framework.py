#!/usr/bin/env python

import argparse
import itertools
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import defs
import exp_settings
from prism import Prism
from utils import get_dataset_path, load_data, save_results
from utils_IEEE_CIS import (
    add_hour_day_month_ieee_cis,
    load_ieee_cis_train_set,
    split_ieee_cis,
)


def store_results(
    metrics, times, metrics_dict, times_dict, idx, run_params_dict, sla_violations=None
):
    for key, value in metrics.items():
        if "total_cost" in key:
            continue
        if key in metrics_dict:
            metrics_dict[key].append(value[idx])

    if sla_violations is None:
        sla_violations = sla_violated(metrics, idx, run_params_dict)
    metrics_dict["total_sla_violations"].append(sla_violations)
    metrics_dict["total_cost"].append(sla_violations * run_params_dict["slaCost"])

    for key, value in times.items():
        if key in times_dict:
            if len(value) > 1:
                times_dict[key].append(value[1])
            else:
                times_dict[key].append(-1)


def count_sla_violations(recall, fpr, run_params_dict):

    print(f"[D] recall: {recall}   fpr: {fpr}")

    count = 0
    if fpr > run_params_dict["fprT"]:
        count += 1
    if recall < run_params_dict["recallT"]:
        count += 1

    return count


def sla_violated_benefits_model(
    _time,
    metrics,
    tmp_dict,
    prism,
    run_params_dict,
):

    tmp_dict["_time"] = _time * tmp_dict["time_interval"]

    metrics["amount_old_data"] = tmp_dict["amount_old_data"]
    metrics["retrain_hour"] = [tmp_dict["last_retrain"] * tmp_dict["time_interval"]]
    metrics["amount_new_data"] = tmp_dict["amount_new_data"]
    metrics["curr_timestamp"] = tmp_dict["curr_timestamp"]

    prism_dict = prism.compute_prism_inputs(
        metrics,
        tmp_dict,
    )

    if "delta" in prism.get_model_type():
        recall = prism_dict["curr_tpr"] + prism_dict["new_TPR_retrain"]
        fpr = 100 - (prism_dict["curr_tnr"] + prism_dict["new_TNR_retrain"])
    else:
        recall = prism_dict["new_TPR_retrain"]
        fpr = 100 - prism_dict["new_TNR_retrain"]

    sla_violations_retrain = count_sla_violations(recall, fpr, run_params_dict)

    sla_violations_nop = -1
    if run_params_dict["nopModels"] is not None:
        if "delta" in prism.get_model_type():
            recall = prism_dict["curr_tpr"] + prism_dict["new_TPR_noRetrain"]
            fpr = 100 - (prism_dict["curr_tnr"] + prism_dict["new_TNR_noRetrain"])
        else:
            recall = prism_dict["new_TPR_noRetrain"]
            fpr = 100 - prism_dict["new_TNR_noRetrain"]

        sla_violations_nop = count_sla_violations(recall, fpr, run_params_dict)

    return sla_violations_nop, sla_violations_retrain


def sla_violated(model, idx, run_params_dict):

    tn, fp, fn, tp = confusion_matrix(  # tn, fp, fn, tp
        model["real_labels"][idx],
        model["predictions"][idx],
    ).ravel()

    recall = (tp / (tp + fn)) * 100
    fpr = (fp / (tn + fp)) * 100

    return count_sla_violations(recall, fpr, run_params_dict)


def update_metrics(
    tmp_dict, run_params_dict, curr_model_metrics, retrain_model_metrics
):
    test_data = run_params_dict["testData"]

    _time = test_data.loc[test_data["timestamp"] >= tmp_dict["curr_timestamp"]][
        "DT_H"
    ].min()

    tx_idx = len(
        test_data.loc[
            (test_data["DT_H"] >= _time)
            & (test_data["DT_H"] < _time + run_params_dict["retrainLatency"])
        ]
    )

    for key in curr_model_metrics.keys():
        # print(key)
        if isinstance(curr_model_metrics[key][0], (list, np.ndarray)):
            aux = curr_model_metrics[key][tmp_dict["_time"]][:tx_idx]
            aux = np.append(aux, retrain_model_metrics[key][tmp_dict["_time"]][tx_idx:])
            curr_model_metrics[key] = retrain_model_metrics[key]
            curr_model_metrics.at[tmp_dict["_time"], key] = aux
        else:
            curr_model_metrics[key] = retrain_model_metrics[key]

    return curr_model_metrics


def check_retrain(
    run_params_dict,
    _time,
    curr_model_metrics,
    retrain_model_metrics,
    metrics_dict,
    tmp_dict,
    prism,
):

    if "random" in run_params_dict["baseline"]:
        if run_params_dict["randomGenerator"].uniform(0, 1) > 0.5:
            tmp_dict["retrain_occurred"] = True
    elif "reactive" in run_params_dict["baseline"]:
        if sla_violated(curr_model_metrics, _time - 1, run_params_dict) > 0:
            tmp_dict["retrain_occurred"] = True
    elif run_params_dict["askPrism"] and (
        "delta" in run_params_dict["baseline"] or "abs" in run_params_dict["baseline"]
    ):
        aux_metrics = metrics_dict.copy()
        aux_tmp_dict = tmp_dict.copy()
        aux_tmp_dict["_time"] = _time * tmp_dict["time_interval"]

        aux_metrics["amount_old_data"] = tmp_dict["amount_old_data"]
        aux_metrics["retrain_hour"] = [
            tmp_dict["last_retrain"] * tmp_dict["time_interval"]
        ]
        aux_metrics["amount_new_data"] = tmp_dict["amount_new_data"]
        aux_metrics["curr_timestamp"] = [tmp_dict["curr_timestamp"]]
        aux_metrics["retrain_timestamp"] = [tmp_dict["last_retrain_timestamp"]]

        time_overhead = time.time()
        tmp_dict["retrain_occurred"] = prism.ask_prism(
            aux_metrics,
            aux_tmp_dict,
        )
        metrics_dict["overall_time_overhead"].append(time.time() - time_overhead)
        print("[D] Going to retrain: " + str(tmp_dict["retrain_occurred"]).upper())
    else:
        tmp_dict["sla_violations"] = sla_violated(
            curr_model_metrics, _time, run_params_dict
        )

        if "optimum" in run_params_dict["baseline"]:
            sla_violations_retrain = sla_violated(
                retrain_model_metrics, _time, run_params_dict
            )
        elif (
            "delta" in run_params_dict["baseline"]
            or "abs" in run_params_dict["baseline"]
        ):
            if run_params_dict["nopModels"] is not None:
                (
                    tmp_dict["sla_violations"],
                    sla_violations_retrain,
                ) = sla_violated_benefits_model(
                    _time, metrics_dict, tmp_dict, prism, run_params_dict
                )
            else:
                _, sla_violations_retrain = sla_violated_benefits_model(
                    _time,
                    metrics_dict,
                    tmp_dict,
                    prism,
                    run_params_dict,
                )

        if tmp_dict["sla_violations"] > 0:  # sla violated if we don’t retrain

            print(
                "[D] violations-NOP="
                + str(tmp_dict["sla_violations"])
                + f"    violations-RET={sla_violations_retrain}"
            )
            if (
                run_params_dict["retrainCost"]
                + run_params_dict["slaCost"] * sla_violations_retrain
                < run_params_dict["slaCost"] * tmp_dict["sla_violations"]
            ):
                tmp_dict["retrain_occurred"] = True
                tmp_dict["sla_violations"] = sla_violations_retrain


def test(
    metrics_dict,
    times_dict,
    datasets_metrics,
    datasets_times,
    run_params_dict,
    test_start_time,
    prism: Prism = None,
):

    time_interval = run_params_dict["timeInterval"]
    baseline = run_params_dict["baseline"]

    curr_model_metrics = datasets_metrics[
        f"timeInterval_{time_interval}-retrainPeriod_{int(test_start_time)}"
    ]
    curr_model_times = datasets_times[
        f"timeInterval_{time_interval}-retrainPeriod_{int(test_start_time)}"
    ]

    if "delta" in baseline or "abs" in baseline or test_start_time != 0:
        test_start_time = test_start_time // time_interval
    _time = test_start_time

    tmp_dict = {
        "last_retrain": _time,
        "len_train": curr_model_times["amount_old_data"].to_numpy()[0],
        "_time": _time,
        "time_interval": time_interval,
        "val_scores": datasets_times[f"timeInterval_{time_interval}-retrainPeriod_0"][
            "val_scores"
        ][0],
        "amount_new_data": [0],
        "amount_old_data": [curr_model_times["amount_old_data"].to_numpy()[0]],
        "amount_new_fraud": [0],
        "curr_new_samples": 0,
        "curr_new_fraud_samples": 0,
        "retrain_occurred": False,
        "sla_violations": None,
        "curr_timestamp": pd.Timestamp(
            curr_model_times["retrain_timestamp"].to_numpy()[-1]
        ),
        "last_retrain_timestamp": pd.Timestamp(
            curr_model_times["retrain_timestamp"].to_numpy()[-1]
        ),
    }

    if "delta" in baseline or "abs" in baseline or _time != 0:
        tmp_dict["amount_new_data"] = [
            curr_model_times["amount_new_data"].to_numpy()[1]
        ]
        tmp_dict["amount_new_fraud"] = [
            curr_model_times["amount_new_fraud"].to_numpy()[1]
        ]

        tmp_dict["retrain_occurred"] = True

        if run_params_dict["retrainLatency"] != 0:
            curr_model_metrics = datasets_metrics[
                f"timeInterval_{time_interval}-retrainPeriod_{(_time-1)*time_interval}"
            ]
            retrain_model_metrics = datasets_metrics[
                f"timeInterval_{time_interval}-retrainPeriod_{_time*time_interval}"
            ]
            curr_model_metrics = update_metrics(
                tmp_dict,
                run_params_dict,
                curr_model_metrics,
                retrain_model_metrics,
            )

    while _time < len(curr_model_metrics["real_labels"]):

        store_results(
            curr_model_metrics,
            curr_model_times,
            metrics_dict,
            times_dict,
            _time,
            run_params_dict,
        )

        tmp_dict["curr_new_samples"] += curr_model_metrics[
            "num_transactions"
        ].to_numpy()[_time]
        tmp_dict["curr_new_fraud_samples"] += curr_model_metrics[
            "num_fraud_transactions"
        ].to_numpy()[_time]

        print(
            f"-------- {baseline}\ttime: {_time}/{len(curr_model_metrics['real_labels'])} -------- "
        )
        _time += 1
        tmp_dict["retrain_occurred"] = False
        tmp_dict["curr_timestamp"] += timedelta(hours=time_interval)
        if _time >= len(curr_model_metrics["real_labels"]):
            break

        tmp_dict["_time"] = _time

        retrain_model_metrics = datasets_metrics[
            f"timeInterval_{time_interval}-retrainPeriod_{_time*time_interval}"
        ]
        retrain_model_times = datasets_times[
            f"timeInterval_{time_interval}-retrainPeriod_{_time*time_interval}"
        ]

        if "periodic" in baseline:
            tmp_dict["retrain_occurred"] = True
        elif "no_retrain" in baseline:
            tmp_dict["retrain_occurred"] = False
        else:
            check_retrain(
                run_params_dict=run_params_dict,
                _time=_time,
                curr_model_metrics=curr_model_metrics,
                retrain_model_metrics=retrain_model_metrics,
                metrics_dict=metrics_dict,
                tmp_dict=tmp_dict,
                prism=prism,
            )

        if tmp_dict["retrain_occurred"]:
            if run_params_dict["retrainLatency"] != 0:
                curr_model_metrics = update_metrics(
                    tmp_dict,
                    run_params_dict,
                    curr_model_metrics,
                    retrain_model_metrics,
                )
            else:
                curr_model_metrics = retrain_model_metrics
            curr_model_times = retrain_model_times
            tmp_dict["curr_timestamp"] = pd.Timestamp(
                curr_model_times["retrain_timestamp"].to_numpy()[-1]
            )
            tmp_dict["last_retrain_timestamp"] = pd.Timestamp(
                curr_model_times["retrain_timestamp"].to_numpy()[-1]
            )
            tmp_dict["last_retrain"] = _time
            tmp_dict["amount_old_data"].append(
                tmp_dict["amount_old_data"][-1] + tmp_dict["amount_new_data"][-1]
            )
            tmp_dict["amount_new_data"].append(tmp_dict["curr_new_samples"])
            tmp_dict["curr_new_samples"] = 0
            tmp_dict["curr_new_fraud_samples"] = 0


def init_prism(run_params_dict: dict):
    time_interval = run_params_dict["timeInterval"]
    dataset_file = f"timeInterval_{time_interval}-rand_sample"
    if "labelShift" in defs.DATASET_NAME_TEST:
        dataset_file = (
            dataset_file + "-" + defs.DATASET_NAME_TEST.split("-", maxsplit=3)[3]
        )

    print(f"[D] Dataset file for RB model: {dataset_file}.pkl")
    prism = Prism(
        time_interval=run_params_dict["timeInterval"],
        model_type=run_params_dict["baseline"].split("_", maxsplit=1)[0],
        dataset_name=defs.DATASET_NAME_TEST,
        seed=1,
        sat_value=run_params_dict["satValue"],
        nop_models=run_params_dict["nopModels"],
        retrain_cost=run_params_dict["retrainCost"],  # cost of the retrain tactic
        retrain_latency=run_params_dict[
            "retrainLatency"
        ],  # how long the retrain tactic takes to execute (in hours)
        fpr_threshold=run_params_dict["fprT"],  # SLA agreed maximum FPR (in %)
        recall_threshold=run_params_dict["recallT"],  # SLA agreed minimum RECALL (in %)
        fpr_sla_cost=run_params_dict["slaCost"],  # cost of violating the FPR SLA
        recall_sla_cost=run_params_dict["slaCost"],  # cost of violating the RECALL SLA
    )

    return prism


def setup_and_test(datasets_metrics, datasets_times, run_params_dict, path):

    time_interval = run_params_dict["timeInterval"]

    test_start_time = 0
    prism = None
    if "delta" in run_params_dict["baseline"] or "abs" in run_params_dict["baseline"]:
        prism = init_prism(run_params_dict)
        test_start_time = prism.get_test_start_time()

    # INITIALIZE RESULTS DICTS
    metrics_dict = {
        "total_sla_violations": [],
        "total_cost": [],
        "retrain_occurred": [],
        "benefits_model_features": [-1],
        "benefits_model_predictions": [-1],
        "benefits_model_target_tpr": [-1],
        "benefits_model_target_tnr": [-1],
        "nop_model_predictions": [-1],
        "prism_time_overhead": [-1],
        "time_series_time_overhead": [-1],
        "time_series_preds": [-1],
        "overall_time_overhead": [-1],
    }
    for key in datasets_metrics[
        f"timeInterval_{time_interval}-retrainPeriod_{int(test_start_time)}"
    ].to_dict():
        if key in [
            "avg_money",
            "count",
            "roc_auc",
            "acc",
            "precisionScore",
            "recallScore",
            "f1Score",
            "loss",
        ]:
            continue
        print(key)
        metrics_dict[key] = []

    times_dict = {}
    for key in datasets_times[
        f"timeInterval_{time_interval}-retrainPeriod_10"
    ].to_dict():
        print(key)
        times_dict[key] = []

    if (
        "delta" in run_params_dict["baseline"]
        or "abs" in run_params_dict["baseline"]
        or test_start_time != 0
    ):
        for i in range(0, test_start_time, time_interval):
            print(i)
            store_results(
                datasets_metrics[
                    f"timeInterval_{time_interval}-retrainPeriod_{int(test_start_time)}"
                ],
                datasets_times[
                    f"timeInterval_{time_interval}-retrainPeriod_{int(test_start_time)}"
                ],
                metrics_dict,
                times_dict,
                i // time_interval,
                run_params_dict,
            )
            metrics_dict["benefits_model_features"].append(-1)
            metrics_dict["benefits_model_predictions"].append(-1)
            metrics_dict["benefits_model_target_tpr"].append(-1)
            metrics_dict["benefits_model_target_tnr"].append(-1)
            metrics_dict["nop_model_predictions"].append(-1)
            metrics_dict["prism_time_overhead"].append(-1)
            metrics_dict["time_series_time_overhead"].append(-1)
            metrics_dict["time_series_preds"].append(-1)
            metrics_dict["overall_time_overhead"].append(-1)

    test(
        metrics_dict,
        times_dict,
        datasets_metrics,
        datasets_times,
        run_params_dict,
        test_start_time,
        prism,
    )

    # SAVE RESULTS
    results_dict = {}
    for key, value in metrics_dict.items():
        print(f"{key}: {len(value)}")
        if len(value) == len(metrics_dict["scores"]):
            results_dict[key] = metrics_dict[key]

    # generate name for the output results file
    results_file = "metrics"
    for key, value in run_params_dict.items():
        if "randomGenerator" not in key and "testData" not in key:
            if "askPrism" in key and "delta" not in run_params_dict["baseline"]:
                value = False
            if "nopModels" in key and "delta" not in run_params_dict["baseline"]:
                value = False
            results_file = results_file + f"-{key}_{value}"
    results_file = results_file + ".pkl"

    print(f"[D] Saving results to {path}tmp/")
    save_results(
        path + "results/files/",
        results_file,
        pd.DataFrame(results_dict),
    )

    return pd.DataFrame(results_dict)


def main(use_pgf: False):
    # pylint: disable=too-many-locals
    test_path = get_dataset_path(defs.DATASET_NAME_TEST)
    path = test_path
    if use_pgf:
        path = path + "pre-generated/"

    for time_interval in defs.TIME_INTERVALS:

        # LOAD DATA FILES
        print(f"[D] Loading data from {test_path}")
        datasets_metrics, datasets_times = load_data(
            path=path,
            time_intervals=[time_interval],
            retrain_periods=list(range(time_interval, 3160, time_interval)),
        )
        print("[D] Finished")

        x_train, y_train = load_ieee_cis_train_set(
            test_path + "original/", defs.DATASET_NAME_TEST
        )

        _, _, _, _, test_data = split_ieee_cis(
            x_train,
            y_train,
            "labels",
            time_interval,
        )
        add_hour_day_month_ieee_cis(test_data)

        # START EVALUATION
        print("[D] Starting testing")

        for seed in range(1, defs.SEEDS + 1):
            random_generator = np.random.RandomState(seed)

            prod = itertools.product(
                defs.SAT_VALUES,
                defs.USE_NOP_MODELS,
                exp_settings.FPR_T,
                exp_settings.RECALL_T,
                exp_settings.SLA_COSTS,
                exp_settings.RETRAIN_COSTS,
                exp_settings.RETRAIN_LATENCIES,
                exp_settings.BASELINES,
            )
            for (
                sat_val,
                nop_models,
                fpr_t,
                recall_t,
                sla_cost,
                retrain_cost,
                retrain_latency,
                baseline,
            ) in prod:

                # SET INITIAL MODEL
                run_params_dict = {
                    "timeInterval": time_interval,
                    "baseline": baseline,
                    "fprT": fpr_t,
                    "recallT": recall_t,
                    "retrainCost": retrain_cost,
                    "retrainLatency": retrain_latency,
                    "slaCost": sla_cost,
                    "satValue": sat_val,
                    "nopModels": nop_models,
                    "askPrism": defs.ASK_PRISM,
                    "seed": seed,
                    "randomGenerator": random_generator,
                    "testData": test_data,
                }
                print("#" * 150)
                print(f"#   {run_params_dict}   #")
                print("#" * 150)

                setup_and_test(
                    datasets_metrics, datasets_times, run_params_dict, test_path
                )


if __name__ == "__main__":

    use_pre_generated_files = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pgf", help="use pre-generated files", action="store_true")
    args = parser.parse_args()

    if args.use_pgf:
        print("[D] Using pre-generated files")
        use_pre_generated_files = True

    main(use_pre_generated_files)
