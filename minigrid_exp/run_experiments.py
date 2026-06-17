import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

from config import (
    ALGORITHM_LABEL,
    ALGORITHM_NAME,
    DQN_EXPLORATION_STRATEGY,
    ENVIRONMENTS,
    LOGS_DIR,
    MODELS_DIR,
    PARALLEL_WORKERS,
    SEEDS,
    TOTAL_TIMESTEPS,
    VARIANTS,
)


_PRINT_LOCK = Lock()


@dataclass(frozen=True)
class ExperimentJob:
    difficulty: str
    env_id: str
    variant_name: str
    intrinsic: bool
    noise: bool
    seed: int
    total_timesteps: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Run configured MiniGrid {ALGORITHM_LABEL} experiments."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=PARALLEL_WORKERS,
        help=(
            "Number of worker threads for running independent experiments. "
            "Use 1 for the old sequential behavior."
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    return args


def build_experiment_jobs():
    jobs = []

    for difficulty, env_ids in ENVIRONMENTS.items():
        for env_id in env_ids:
            for variant in VARIANTS:
                for seed in SEEDS:
                    jobs.append(
                        ExperimentJob(
                            difficulty=difficulty,
                            env_id=env_id,
                            variant_name=variant["name"],
                            intrinsic=variant["intrinsic"],
                            noise=variant["noise"],
                            seed=seed,
                            total_timesteps=TOTAL_TIMESTEPS,
                        )
                    )

    return jobs


def print_job_header(job: ExperimentJob, status: str):
    with _PRINT_LOCK:
        print("=" * 100)
        print(f"{status}: {job.env_id} | {job.variant_name} | seed {job.seed}")
        print(f"Algorithm: {ALGORITHM_LABEL}")
        if ALGORITHM_NAME == "dqn":
            print(f"DQN exploration: {DQN_EXPLORATION_STRATEGY}")
        print(f"Difficulty: {job.difficulty}")
        print(f"Environment: {job.env_id}")
        print(f"Variant: {job.variant_name}")
        print(f"Intrinsic reward: {job.intrinsic}")
        print(f"Noise: {job.noise}")
        print(f"Seed: {job.seed}")
        print("=" * 100)


def print_job_completed(job: ExperimentJob, model_path):
    with _PRINT_LOCK:
        print("=" * 100)
        print(f"Completed: {job.env_id} | {job.variant_name} | seed {job.seed}")
        print(f"Saved model: {model_path}")
        print("=" * 100)


def print_job_failed(job: ExperimentJob, error: Exception):
    with _PRINT_LOCK:
        print("=" * 100)
        print(f"Failed: {job.env_id} | {job.variant_name} | seed {job.seed}")
        print(f"Reason: {error}")
        print("=" * 100)


def run_experiment(job: ExperimentJob):
    from train import train_agent

    print_job_header(job, "Starting")

    return train_agent(
        env_id=job.env_id,
        variant_name=job.variant_name,
        intrinsic=job.intrinsic,
        noise=job.noise,
        seed=job.seed,
        total_timesteps=job.total_timesteps,
        log_dir=LOGS_DIR,
        model_dir=MODELS_DIR,
    )


def run_jobs(jobs, max_workers: int):
    if not jobs:
        print("No experiments configured.")
        return

    max_workers = min(max_workers, len(jobs))
    print(
        f"Running {len(jobs)} experiment(s) "
        f"with {max_workers} worker thread(s)."
    )

    if max_workers == 1:
        for job in jobs:
            model_path = run_experiment(job)
            print_job_completed(job, model_path)
        return

    failures = []

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="experiment",
    ) as executor:
        future_to_job = {
            executor.submit(run_experiment, job): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]

            try:
                model_path = future.result()
            except Exception as error:
                failures.append((job, error))
                print_job_failed(job, error)
            else:
                print_job_completed(job, model_path)

    if failures:
        raise RuntimeError(f"{len(failures)} experiment(s) failed.")


def main():
    args = parse_args()
    jobs = build_experiment_jobs()
    run_jobs(jobs, args.workers)


if __name__ == "__main__":
    main()
