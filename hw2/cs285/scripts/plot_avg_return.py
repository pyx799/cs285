import argparse
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def load_scalars(logdir, scalar_name):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    if scalar_name not in ea.Tags()['scalars']:
        raise ValueError(f"Scalar '{scalar_name}' not found in {logdir}. Available scalars: {ea.Tags()['scalars']}")

    events = ea.Scalars(scalar_name)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    return steps, values


def main(args):
    # 加载两个 scalar
    steps_x, values_x = load_scalars(args.logdir, args.scalar_x)
    steps_y, values_y = load_scalars(args.logdir, args.scalar_y)

    # 检查 step 是否一致
    if steps_x != steps_y:
        print("⚠️ Warning: Steps are not aligned between the two scalars.")
        print("Proceeding with matching by step.")

        # 尝试通过 step 对齐
        step_to_value_x = dict(zip(steps_x, values_x))
        step_to_value_y = dict(zip(steps_y, values_y))

        common_steps = sorted(set(steps_x) & set(steps_y))

        values_x = [step_to_value_x[s] for s in common_steps]
        values_y = [step_to_value_y[s] for s in common_steps]
    else:
        common_steps = steps_x

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(values_x, values_y, marker='o')
    plt.xlabel(args.scalar_x)
    plt.ylabel(args.scalar_y)
    plt.title(f"{args.scalar_y} vs {args.scalar_x}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot one scalar against another from TensorBoard logs.")
    parser.add_argument('--logdir', type=str, required=True, help="Path to TensorBoard log directory")
    parser.add_argument('--scalar_x', type=str, default='Train_EnvstepsSoFar', help="Scalar name for x-axis")
    parser.add_argument('--scalar_y', type=str, default='Train_AverageReturn', help="Scalar name for y-axis")

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        raise FileNotFoundError(f"Log directory {args.logdir} does not exist.")

    main(args)