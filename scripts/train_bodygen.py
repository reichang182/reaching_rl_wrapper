"""Run BodyGen training on the co_design_task environment."""
import argparse
import os
import subprocess
import sys
import types


def ensure_co_design_task(repo_base: str = "bodygen/co_design_task_repo") -> None:
    """Ensure :mod:`co_design_task` can be imported.

    If the package is missing, clone it from GitHub and install it in editable
    mode. The upstream repository uses a placeholder project name, so we patch
    ``pyproject.toml`` to use the correct package name before installation.
    """

    try:  # pragma: no cover - optional dependency
        import co_design_task  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    repo_dir = os.path.abspath(repo_base)
    if not os.path.exists(repo_dir):
        subprocess.check_call(
            [
                "git",
                "clone",
                "https://github.com/reichang182/co_design_task",
                repo_dir,
            ]
        )

    pyproj = os.path.join(repo_dir, "pyproject.toml")
    if os.path.exists(pyproj):
        with open(pyproj, "r") as f:
            text = f.read()
        if "YOUR_PROJECT" in text:
            text = text.replace("YOUR_PROJECT", "co_design_task")
            with open(pyproj, "w") as f:
                f.write(text)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", repo_dir])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BodyGen on IK task")
    parser.add_argument(
        "--bodygen-dir", default="bodygen/repo", help="Path to clone BodyGen repository"
    )
    parser.add_argument("--cfg", default="cheetah", help="Base BodyGen config name")
    parser.add_argument("--dof", type=int, default=3, help="Degrees of freedom")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Additional Hydra overrides")
    args = parser.parse_args()

    # Ensure co_design_task is installed before launching BodyGen
    ensure_co_design_task()

    repo_dir = os.path.abspath(args.bodygen_dir)
    if not os.path.exists(repo_dir):
        subprocess.check_call([
            "git",
            "clone",
            "https://github.com/Josh00-Lu/BodyGen",
            repo_dir,
        ])

    # Provide a lightweight cv2 stub if OpenCV is missing
    try:  # pragma: no cover - runtime dependency check
        import cv2  # type: ignore
    except Exception:
        stub_src = os.path.join(repo_dir, "cv2.py")
        with open(stub_src, "w") as f:
            f.write(
                "COLOR_RGB2BGRA=0\nCOLOR_RGB2BGR=0\n"
                "def cvtColor(img, code):\n    return img\n"
                "def imwrite(*args, **kw):\n    pass\n"
            )
        stub_mod = types.ModuleType("cv2")
        stub_mod.COLOR_RGB2BGRA = 0
        stub_mod.COLOR_RGB2BGR = 0
        stub_mod.cvtColor = lambda img, code: img
        stub_mod.imwrite = lambda *a, **kw: None
        sys.modules["cv2"] = stub_mod

    # Overwrite BodyGen env registry to avoid mujoco imports
    env_init = os.path.join(repo_dir, "design_opt", "envs", "__init__.py")
    with open(env_init, "w") as f:
        f.write("from bodygen.envs import IKBodyGenEnv\n")
        f.write("env_dict = {'co_design_ik': IKBodyGenEnv}\n")

    # Force BodyGen config to use our environment
    cfg_file = os.path.join(repo_dir, "design_opt", "cfg", f"{args.cfg}.yml")
    if os.path.exists(cfg_file):
        lines = []
        with open(cfg_file) as f:
            for line in f:
                if line.startswith("env_name:"):
                    line = "env_name: co_design_ik\n"
                lines.append(line)
        with open(cfg_file, "w") as f:
            f.writelines(lines)
        # Append env_specs block if missing
        if not any(l.startswith("env_specs:") for l in lines):
            lines.append("env_specs:\n")
            lines.append(f"  dof: {args.dof}\n")
            lines.append(f"  number_of_targets: 1\n")
            with open(cfg_file, "w") as f:
                f.writelines(lines)

    sys.path.insert(0, repo_dir)
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Import env_dict from BodyGen but fall back to an empty dict if optional
    # dependencies (like mujoco) are missing.
    try:
        from design_opt.envs import env_dict
    except Exception as e:  # pragma: no cover - best effort import
        print("Warning: failed to import BodyGen envs:", e)
        env_dict = {}
    # Import our wrapper environment
    from bodygen.envs import IKBodyGenEnv

    # Register our environment
    env_dict["co_design_ik"] = IKBodyGenEnv

    overrides = [f"cfg={args.cfg}"]
    if args.extra:
        overrides.extend(args.extra)

    cmd = [sys.executable, "-m", "design_opt.train", *overrides]
    env = os.environ.copy()
    extra_paths = [repo_dir, os.path.dirname(os.path.dirname(__file__))]
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + [env.get("PYTHONPATH", "")])
    subprocess.run(cmd, cwd=repo_dir, check=True, env=env)


if __name__ == "__main__":
    main()
