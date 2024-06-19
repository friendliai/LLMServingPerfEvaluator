# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-lines,missing-function-docstring,fixme,broad-exception-caught,too-many-locals,too-many-arguments,super-with-arguments

"""FriendliAI LLM workload generator."""

import argparse
import asyncio
import datetime
import os
import random
import time
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import uvloop
from prometheus_client import Gauge, start_http_server
from prometheus_summary import Summary
from transformers import AutoTokenizer  # type: ignore

from engine_client import get_engine_client_factory, get_request_config
from engine_client.base import ClientType, CompletionResult, Engine, EngineClient
from workload import get_workload

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def get_stats_from_result(
    result: pd.Series,
) -> Tuple:  # pylint: disable=too-many-instance-attributes
    if len(result) == 0:
        return (0, 0, 0, 0, 0, 0, 0)
    return (
        result.mean(),
        np.percentile(result, 25),
        np.percentile(result, 50),
        np.percentile(result, 75),
        np.percentile(result, 90),
        np.percentile(result, 95),
        np.percentile(result, 99),
    )


def export_result(
    result_path: Path,
    results: List[CompletionResult],
    request_rate: float,
    total_time: float,
    stream: bool = False,
    verbose: bool = False,
):
    latency = pd.Series([result.end_time - result.start_time for result in results])
    prompt_length = pd.Series([result.prompt_length for result in results])
    response_length = pd.Series([result.response_length for result in results])
    latency_per_output_token = latency / response_length * 1000

    latency_stats = get_stats_from_result(latency)
    token_latency_stats = get_stats_from_result(latency_per_output_token)

    data = {
        "exp_time": [datetime.datetime.now()],
        "lambda": request_rate,
        "total_time": total_time,
        "thp": len(latency) / total_time,
        "thp in tokens": prompt_length.sum() / total_time,
        "thp out tokens": response_length.sum() / total_time,
        # latency
        "mean lat": latency_stats[0],
        "p25 lat": latency_stats[1],
        "p50 lat": latency_stats[2],
        "p75 lat": latency_stats[3],
        "p90 lat": latency_stats[4],
        "p95 lat": latency_stats[5],
        "p99 lat": latency_stats[6],
        # token latency
        "mean token lat": token_latency_stats[0],
        "p25 token lat": token_latency_stats[1],
        "p50 token lat": token_latency_stats[2],
        "p75 token lat": token_latency_stats[3],
        "p90 token lat": token_latency_stats[4],
        "p95 token lat": token_latency_stats[5],
        "p99 token lat": token_latency_stats[6],
    }

    if verbose:
        print(f"Total time: {total_time:.3f} s")
        print(
            f"Throughput: {len(latency) / total_time:.3f} requests/s, "
            f"{prompt_length.sum() / total_time:.3f} input tokens/s, "
            f"{response_length.sum() / total_time:.3f} output tokens/s"
        )
        print(
            f"Latency: mean = {latency_stats[0]:.3f} s; p90 = {latency_stats[4]:.3f} s; "
            f"p99 = {latency_stats[6]:.3f} s"
        )
        print(
            f"Latency per output token: mean = {token_latency_stats[0]:.3f} ms; "
            f"p90 = {token_latency_stats[4]:.3f} ms; "
            f"p99 = {token_latency_stats[6]:.3f} ms"
        )

    if stream:
        time_to_first_token = (
            pd.Series(
                [
                    result.first_token_end_time - result.start_time  # type: ignore
                    for result in results
                ]
            )
            * 1000
        )
        latency_except_first_token = pd.Series(
            [result.end_time - result.first_token_end_time for result in results]  # type: ignore
        )
        time_per_output_token = (
            latency_except_first_token / (response_length - 1) * 1000
        )
        time_per_output_token = time_per_output_token.replace(  # type: ignore
            [-np.inf, np.inf], np.nan
        ).dropna()  # drop response_length is equal to 1

        ttft_stats = get_stats_from_result(time_to_first_token)
        tpot_stats = get_stats_from_result(time_per_output_token)

        # TTFT
        data["mean TTFT"] = ttft_stats[0]
        data["p25 TTFT"] = ttft_stats[1]
        data["p50 TTFT"] = ttft_stats[2]
        data["p75 TTFT"] = ttft_stats[3]
        data["p90 TTFT"] = ttft_stats[4]
        data["p95 TTFT"] = ttft_stats[5]
        data["p99 TTFT"] = ttft_stats[6]

        # TPOT
        data["mean TPOT"] = tpot_stats[0]
        data["p25 TPOT"] = tpot_stats[1]
        data["p50 TPOT"] = tpot_stats[2]
        data["p75 TPOT"] = tpot_stats[3]
        data["p90 TPOT"] = tpot_stats[4]
        data["p95 TPOT"] = tpot_stats[5]
        data["p99 TPOT"] = tpot_stats[6]

        if verbose:
            print(
                f"Time to first token: mean = {ttft_stats[0]:.3f} ms; "
                f"p90 = {ttft_stats[4]:.3f} ms; "
                f"p99 = {ttft_stats[6]:.3f} ms"
            )
            print(
                f"Time per output token: mean = {tpot_stats[0]:.3f} ms; "
                f"p90 = {tpot_stats[4]:.3f} ms; "
                f"p99 = {tpot_stats[6]:.3f} ms"
            )

    data_frame = pd.DataFrame(data)
    data_frame.to_csv(
        result_path, mode="a", index=False, header=not os.path.isfile(result_path)
    )


def calculate_stats(
    args,
    exp_queues: List[Queue],
    metrics_host: str,
    metrics_port: int,
    stream: bool,
    reqeust_rate: float,
):
    total_responses = Gauge("total_responses", "Total number of responses")
    total_output_tokens = Gauge("total_output_tokens", "Total number of output tokens")
    total_input_tokens = Gauge("total_input_tokens", "Total number of input tokens")
    request_latencies = Summary(
        "request_latencies",
        "Latencies of requests, in microseconds",
        max_age_seconds=args.time_window,
        age_buckets=args.num_age_buckets,
    )
    token_latencies = Summary(
        "token_latencies",
        "Latencies of requests per output token, in microseconds",
        max_age_seconds=args.time_window,
        age_buckets=args.num_age_buckets,
    )
    if stream:
        time_to_first_tokens = Summary(
            "time_to_first_tokens",
            "Time to first token",
            max_age_seconds=args.time_window,
            age_buckets=args.num_age_buckets,
        )
        time_per_output_tokens = Summary(
            "time_per_output_tokens",
            "Time per output token",
            max_age_seconds=args.time_window,
            age_buckets=args.num_age_buckets,
        )
    exp_info = Gauge("experiment_info", "Request rate")

    start_http_server(port=metrics_port, addr=metrics_host)
    exp_info.set(reqeust_rate)
    while True:
        for queue in exp_queues:
            queue.put(None)
        for queue in exp_queues:
            for result in iter(queue.get, None):
                total_responses.inc()
                total_output_tokens.inc(result.response_length)
                total_input_tokens.inc(result.prompt_length)
                request_latencies.observe(result.end_time - result.start_time)
                token_latencies.observe(
                    (result.end_time - result.start_time)
                    * 1000
                    / result.response_length
                )
                if stream:
                    time_per_output_tokens.observe(
                        (result.end_time - result.first_token_end_time)
                        * 1000
                        / (result.response_length - 1)
                    )
                    time_to_first_tokens.observe(
                        (result.first_token_end_time - result.start_time) * 1000
                    )


def main(args):  # pylint: disable=too-many-branches,too-many-statements
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_prefix: str = f"{args.engine}_{args.model_name.split('/')[-1]}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    request_rate_list = sorted([float(r) for r in args.request_rate.split(",")])
    for request_rate in request_rate_list:
        request_config = get_request_config(args.engine, args.request_config_path)
        workload = get_workload(
            args.workload_config_path, args.duration * request_rate, tokenizer
        )
        engine_client_factory = get_engine_client_factory(args.engine)
        engine_clients: List[EngineClient] = [
            engine_client_factory(
                pid,
                args.host,
                args.port,
                args.num_proc,
                args.use_token,
                workload.warmup_size,
                tokenizer,
                workload.dataset,
                request_config,
                args.client_type,
            )
            for pid in range(args.num_proc)
        ]
        engine_clients[0].check_health()

        # Store experiment args and config
        print(f"Start exp for request rate({request_rate})")
        config_path = args.output_dir / f"{output_prefix}.conf"

        with open(config_path, "a+", encoding="utf-8") as f:
            f.write(f"request rate: {request_rate}\n")
            f.write(str(args) + "\n")
            f.write(str(workload.workload_config) + "\n")
            f.write(str(request_config.model_dump()) + "\n")

        engine_clients[0].warm_up()
        if args.verbose:
            print(f"Warm up finished for request rate({request_rate})")

        with Manager() as process_manager:
            return_dict = process_manager.dict()
            exp_queues = [Queue() for _ in range(args.num_proc)]
            try:
                writer_process = Process(
                    target=calculate_stats,
                    args=(
                        args,
                        exp_queues,
                        args.metrics_host,
                        args.metrics_port,
                        request_config.stream,
                        request_rate,
                    ),
                )
                exp_processes = [
                    Process(
                        target=client.wrapper_async_run,
                        args=(
                            return_dict,
                            exp_queues[i],
                            request_rate / args.num_proc,
                        ),
                    )
                    for i, client in enumerate(engine_clients)
                ]
                writer_process.start()
                start = time.time()
                for p in exp_processes:
                    p.start()
                for p in exp_processes:
                    p.join()
                writer_process.terminate()
                writer_process.join()
                results = []
                for res in return_dict.values():
                    results += res
            except Exception as e:
                print(f"Exception occurred: {str(e)}")
                break
            end = time.time()

        csv_path = args.output_dir / f"{output_prefix}.csv"
        export_result(
            csv_path,
            results,
            request_rate,
            end - start,
            request_config.stream,
            args.verbose,
        )
        print(f"Finished exp for request rate({request_rate})")

        if (end - start) > args.timeout:
            print(f"Finish exp due to timeout({args.timeout}s)")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=Engine,
        default=Engine.FRIENDLI,
        choices=[Engine.FRIENDLI, Engine.VLLM],
    )
    parser.add_argument(
        "--client-type",
        type=ClientType,
        default=ClientType.HTTP,
        choices=[ClientType.HTTP, ClientType.GRPC],
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int)
    parser.add_argument("--metrics-host", type=str, default="localhost")
    parser.add_argument("--metrics-port", type=int, default=8000)
    parser.add_argument(
        "--num-proc", type=int, default=4, help="Number of processes to launch"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name or path of the tokenizer in huggingface. "
        "For vllm, model name used to build request body"
        "(e.g., meta-llama/Llama-2-7b-chat-hf)",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=10,
        help="Time window for calculating quantile of metrics in seconds.",
    )
    parser.add_argument(
        "--num-age-buckets",
        type=int,
        default=2,
        help="Time window for calculating quantile of metrics in seconds.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration of the experiment in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout of the experiment in seconds",
    )
    parser.add_argument(
        "--request-rate",
        type=str,
        required=True,
        help=(
            "Comma separated numbers of requests per second"
            " (i.e., lambda of the Poisson distribution)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--workload-config-path",
        type=Path,
        required=True,
        help="Path of the input/output len sampler config file.",
    )
    parser.add_argument(
        "--request-config-path",
        type=Path,
        help="Path of the request config file.",
    )
    parser.add_argument(
        "--use-token",
        action="store_true",
        help="Use tokens instead of strings for request.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print verbose stats",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save results"
    )

    main(parser.parse_args())
