"""LLM-based optimizer for InverseKinematicsTask morphology optimization.

Uses an LLM to propose robot configurations and iteratively refine them based on task performance.
"""

import argparse
import ast
import json
import os
import sys
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add co_design_task to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import warp as wp

# Import from co_design_task
from co_design_task.context.inverse_kinematics_context_manager import (
    InverseKinematicsContextManager,
)
from co_design_task.task.inverse_kinematics_task import InverseKinematicsTask

# LLM imports (optional - will use dummy if not available)
try:
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnableWithMessageHistory
    from langchain_google_genai import ChatGoogleGenerativeAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Using dummy LLM for demonstration.")

# Try to import weave for experiment tracking (optional)
try:
    import weave

    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    print("Warning: Weave not available. Experiment tracking disabled.")


class DummyLLM:
    """Dummy LLM for when LangChain is not available."""

    def invoke(self, messages):
        # Generate random configurations as a placeholder
        dof = 3  # Assuming default DOF
        n_configs = 5
        configs = []
        for _ in range(n_configs):
            config = []
            # Joint angles
            for _ in range(dof):
                config.append(np.random.uniform(-np.pi, np.pi))
            # Link lengths
            for _ in range(dof):
                config.append(np.random.uniform(0.05, 0.3))
            configs.append(config)

        class DummyResponse:
            def __init__(self, content):
                self.content = content

        return DummyResponse(json.dumps(configs))


# Per-session chat history store
_session_histories = {}


def get_chat_history(session_id: str):
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def initialize_task(
    seed: int,
    dof: int,
    num_targets: int,
    enable_obstacles: bool = False,
    num_obstacles: int = 0,
) -> InverseKinematicsTask:
    """Initialize the InverseKinematicsTask environment."""
    wp.rand_init(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create configuration for InverseKinematicsTask
    config = {
        "general": {
            "stage_path": "./llm_optimization_outputs",
            "random_seed": seed,
            "device": "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu",
        },
        "simulation": {"integrator_type": "euler", "sim_substeps": 5, "fps": 30},
        "task": {
            "degrees_of_freedom": dof,
            "number_of_targets": num_targets,
            "number_of_instances": 1,
            "grid_offset": 1.0,
            "table_size_x": 1.0,
            "table_size_y": 0.01,
            "table_size_z": 1.0,
            "robot_base_height": 0.05,
            "default_link_length": 0.15,
            "default_link_radius": 0.02,
            "end_effector_sphere_radius": 0.02,
            "enable_obstacles": enable_obstacles,
            "number_of_obstacles": num_obstacles,
            "requires_grad": False,
            "rigid_contact_margin": 0.01,
            "target_lower_bounds": [-0.5, 0.0, -0.5],
            "target_upper_bounds": [0.5, 0.5, 0.5],
            "joint_reset_lower": -np.pi,
            "joint_reset_upper": np.pi,
            # Additional required parameters
            "target_shared_random_position": False,
            "joint_pose_lower_bounds": [-np.pi] * dof,
            "joint_pose_upper_bounds": [np.pi] * dof,
            "link_length_reset_handler": "static",
            "link_length_static_values": [0.15] * dof,
            "collision": True,
            "collision_weight": 1.0,
        },
    }

    # Create task using context manager
    task = InverseKinematicsContextManager.create(config)

    # Reset to get initial state
    task.reset()

    return task


def evaluate_configuration(
    task: InverseKinematicsTask,
    config: List[float],
    dof: int,
    num_targets: int,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate a single configuration and return (loss, metrics).
    Config format: [j1, j2, ..., jn, l1, l2, ..., ln] where j=joint angles, l=link lengths
    """
    if len(config) != 2 * dof:
        print(f"Warning: Invalid configuration length {len(config)}, expected {2*dof}")
        return float("inf"), {"error": "invalid_config_length"}

    joint_angles = np.array(config[:dof], dtype=np.float32)
    link_lengths = np.array(config[dof:], dtype=np.float32)

    # Clamp values to valid ranges
    joint_angles = np.clip(joint_angles, -np.pi, np.pi)
    link_lengths = np.clip(link_lengths, 0.05, 0.3)

    try:
        # Reset task to clear previous state
        task.reset()

        # Set link lengths by modifying the feature states
        feature_states = task.get_feature_states()
        if "link_lengths" in feature_states:
            task.feature_states["link_lengths"].numpy()[:dof] = link_lengths

        # Apply joint angles (repeat for each target if multiple)
        joint_poses_all = np.tile(joint_angles, num_targets)
        task.feature_states["joint_poses"].numpy()[:] = joint_poses_all

        # Step the task
        feature_states = task.step()

        # Get results
        feature_states = task.get_feature_states()

        # Calculate metrics
        total_distance = 0.0
        for target_idx in range(num_targets):
            ee_pos = feature_states["end_effector_position"][target_idx]
            target_pos = feature_states["target_position"][target_idx]
            distance = np.linalg.norm(ee_pos - target_pos)
            total_distance += distance

        avg_distance = total_distance / num_targets
        machine_cost = feature_states["machine_cost"][0]

        # Check collisions
        collision_count = 0
        if "collisions" in feature_states:
            collision_count = np.sum(feature_states["collisions"])

        # Calculate combined loss for optimization
        loss = avg_distance + 0.01 * machine_cost  # Simple weighted sum

        metrics = {
            "loss": loss,
            "avg_distance": avg_distance,
            "machine_cost": machine_cost,
            "collision_count": collision_count,
            "joint_angles": joint_angles.tolist(),
            "link_lengths": link_lengths.tolist(),
        }

        return loss, metrics

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return float("inf"), {"error": str(e)}


def parse_llm_response(response_content: str, dof: int, num_configs: int) -> List[List[float]]:
    """Parse LLM's response into a list of configurations."""
    try:
        # Remove markdown code blocks if present
        if response_content.strip().startswith("```json"):
            response_content = response_content.strip()[7:-3].strip()
        elif response_content.strip().startswith("```"):
            response_content = response_content.strip()[3:-3].strip()

        configs = json.loads(response_content)

        if not isinstance(configs, list):
            print("Warning: LLM response is not a list")
            return []

        valid_configs = []
        for i, cfg in enumerate(configs):
            if isinstance(cfg, list) and len(cfg) == 2 * dof:
                try:
                    float_cfg = [float(x) for x in cfg]
                    valid_configs.append(float_cfg)
                except (ValueError, TypeError):
                    print(f"Warning: Config {i} has non-numeric elements. Skipping.")
            else:
                print(f"Warning: Invalid config format from LLM. Expected list of length {2*dof}")

        return valid_configs[:num_configs]  # Return at most num_configs

    except json.JSONDecodeError:
        print("Warning: LLM response was not valid JSON")
        # Try ast.literal_eval as fallback
        try:
            configs = ast.literal_eval(response_content)
            if isinstance(configs, list):
                return configs[:num_configs]
        except:
            pass
        return []


def format_feedback_for_llm(evaluated_configs: List[Dict[str, float]]) -> str:
    """Format evaluation results for the LLM."""
    if not evaluated_configs:
        return "No valid configurations were evaluated."

    feedback = "Results from your previous suggestions:\n\n"
    for i, metrics in enumerate(evaluated_configs):
        feedback += f"{i+1}. Configuration:\n"
        feedback += f"   Joint angles: {metrics.get('joint_angles', 'N/A')}\n"
        feedback += f"   Link lengths: {metrics.get('link_lengths', 'N/A')}\n"
        feedback += "   Results:\n"
        feedback += f"   - Loss: {metrics.get('loss', 'N/A'):.4f}\n"
        feedback += f"   - Avg Distance: {metrics.get('avg_distance', 'N/A'):.4f}\n"
        feedback += f"   - Machine Cost: {metrics.get('machine_cost', 'N/A'):.4f}\n"
        feedback += f"   - Collisions: {metrics.get('collision_count', 0)}\n\n"

    feedback += "Please analyze these results and suggest improved configurations."
    return feedback


def main():
    parser = argparse.ArgumentParser(description="LLM-based optimizer for InverseKinematicsTask")
    parser.add_argument("--dof", type=int, default=3, help="Degrees of freedom")
    parser.add_argument("--num_targets", type=int, default=1, help="Number of targets")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of LLM refinement iterations"
    )
    parser.add_argument(
        "--configs_per_iteration", type=int, default=5, help="Configurations per LLM call"
    )
    parser.add_argument("--enable_obstacles", action="store_true", help="Enable obstacles")
    parser.add_argument("--num_obstacles", type=int, default=0, help="Number of obstacles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--machine_cost_weight", type=float, default=0.01, help="Weight for machine cost"
    )
    parser.add_argument("--llm_model", type=str, default="gemini-1.5-pro", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")

    args = parser.parse_args()

    # Initialize weave if available
    if WEAVE_AVAILABLE:
        weave.init("ik_llm_optimizer")

    print("Initializing InverseKinematicsTask...")
    task = initialize_task(
        seed=args.seed,
        dof=args.dof,
        num_targets=args.num_targets,
        enable_obstacles=args.enable_obstacles,
        num_obstacles=args.num_obstacles,
    )

    # Get initial target positions
    feature_states = task.get_feature_states()
    target_positions = feature_states["target_position"]
    print(f"Target positions: {target_positions}")

    # Initialize LLM
    if LANGCHAIN_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        llm = ChatGoogleGenerativeAI(
            model=args.llm_model,
            temperature=args.temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        print("Using dummy LLM (set GOOGLE_API_KEY environment variable to use real LLM)")
        llm = DummyLLM()

    # Create prompt template
    system_prompt = f"""You are an expert robotics engineer optimizing a {args.dof}-DOF robot arm for inverse kinematics.

Task: Find joint angles and link lengths that position the end-effector at the target(s).

Configuration format: [joint1, joint2, ..., joint{args.dof}, link1, link2, ..., link{args.dof}]

Constraints:
- Joint angles: -π to π radians (-3.14 to 3.14)
- Link lengths: 0.05 to 0.3 meters
- Minimize: distance to target + {args.machine_cost_weight} * machine_cost
- Avoid collisions

Target positions: {target_positions.tolist()}

Respond with EXACTLY {args.configs_per_iteration} configurations as a JSON list of lists.
Example: [[0.5, -0.3, 1.2, 0.15, 0.2, 0.15], [0.6, -0.2, 1.0, 0.1, 0.25, 0.2]]"""

    if LANGCHAIN_AVAILABLE:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        chain = RunnableWithMessageHistory(
            prompt_template | llm,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    else:
        chain = llm  # Use dummy LLM directly

    # Optimization loop
    best_config = None
    best_loss = float("inf")
    best_metrics = {}

    session_id = f"session_{args.seed}"

    for iteration in range(args.iterations):
        print(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")

        # Prepare input for LLM
        if iteration == 0:
            user_input = "Please suggest initial configurations for the robot."
        else:
            user_input = format_feedback_for_llm(evaluated_configs)

        # Get LLM suggestions
        if LANGCHAIN_AVAILABLE:
            response = chain.invoke(
                {"input": user_input}, config={"configurable": {"session_id": session_id}}
            )
            llm_configs = parse_llm_response(response.content, args.dof, args.configs_per_iteration)
        else:
            response = chain.invoke(None)
            llm_configs = parse_llm_response(response.content, args.dof, args.configs_per_iteration)

        print(f"LLM suggested {len(llm_configs)} configurations")

        # Evaluate configurations
        evaluated_configs = []
        for i, config in enumerate(llm_configs):
            loss, metrics = evaluate_configuration(task, config, args.dof, args.num_targets)
            evaluated_configs.append(metrics)

            print(
                f"  Config {i+1}: Loss={loss:.4f}, Distance={metrics.get('avg_distance', 'N/A'):.4f}"
            )

            # Track best configuration
            if loss < best_loss:
                best_loss = loss
                best_config = config
                best_metrics = metrics

        # Log to weave if available
        if WEAVE_AVAILABLE:
            weave.log(
                {
                    "iteration": iteration,
                    "configs_evaluated": len(evaluated_configs),
                    "best_loss": best_loss,
                    "best_distance": best_metrics.get("avg_distance", None),
                }
            )

    # Print final results
    print("\n=== Optimization Complete ===")
    print("Best configuration found:")
    print(f"  Joint angles: {best_metrics.get('joint_angles', 'N/A')}")
    print(f"  Link lengths: {best_metrics.get('link_lengths', 'N/A')}")
    print(f"  Loss: {best_loss:.4f}")
    print(f"  Distance: {best_metrics.get('avg_distance', 'N/A'):.4f}")
    print(f"  Machine cost: {best_metrics.get('machine_cost', 'N/A'):.4f}")
    print(f"  Collisions: {best_metrics.get('collision_count', 0)}")

    # Save best configuration
    output_dir = "./llm_optimization_outputs"
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "args": vars(args),
        "best_config": best_config,
        "best_metrics": best_metrics,
        "target_positions": target_positions.tolist(),
    }

    output_file = os.path.join(output_dir, f"best_config_dof{args.dof}_seed{args.seed}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBest configuration saved to: {output_file}")


if __name__ == "__main__":
    main()
