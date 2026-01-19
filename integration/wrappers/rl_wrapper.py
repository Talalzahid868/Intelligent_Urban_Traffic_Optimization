"""
RL Module Wrapper

Provides a standardized interface to the Reinforcement Learning (Q-Learning)
module (module5_RL) for traffic signal optimization.
"""

import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class ActionResult:
    """Result from RL policy action selection."""
    action_id: int
    action_name: str
    expected_reward: float
    state: Tuple[int, int]
    q_values: Dict[int, float] = None
    
    def to_dict(self) -> dict:
        return {
            'action_id': self.action_id,
            'action_name': self.action_name,
            'expected_reward': self.expected_reward,
            'state': self.state
        }


class RLWrapper:
    """
    Wrapper for RL-based traffic signal optimization.
    
    This wrapper provides a clean interface to the Q-learning policy
    trained in module5_RL for optimal traffic signal control.
    
    Example:
        wrapper = RLWrapper()
        # State: (vehicle_level, pedestrian_level) - discretized
        result = wrapper.get_action(state=(1, 0))  # Medium vehicles, low pedestrians
        print(f"Recommended action: {result.action_name}")
    """
    
    # Action names
    ACTION_NAMES = [
        "Short Green (10s)",
        "Medium Green (20s)",
        "Long Green (30s)",
        "Emergency Mode"
    ]
    
    def __init__(self, policy_path: str = None):
        """
        Initialize the RL wrapper.
        
        Args:
            policy_path: Path to saved Q-table policy (pickle format).
        """
        self._q_table: Dict[Tuple, np.ndarray] = None
        self._policy_path = policy_path
        self.n_actions = len(self.ACTION_NAMES)
    
    def _load_policy(self):
        """Lazy load the Q-table policy."""
        if self._q_table is None:
            if self._policy_path:
                policy_path = Path(self._policy_path)
            else:
                policy_path = Path(__file__).parent.parent.parent / "module5_RL" / "traffic_qlearning_policy.pkl"
            
            if policy_path.exists():
                try:
                    with open(policy_path, 'rb') as f:
                        self._q_table = pickle.load(f)
                except Exception as e:
                    print(f"Warning: Could not load policy from {policy_path}: {e}")
                    self._init_default_policy()
            else:
                print(f"Warning: Policy not found at {policy_path}, using default policy")
                self._init_default_policy()
    
    def _init_default_policy(self):
        """Initialize a default/heuristic policy."""
        # Simple heuristic policy based on traffic conditions
        self._q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        # Define some sensible defaults
        # State: (vehicle_level, pedestrian_level) where level is 0=low, 1=medium, 2=high
        for v in range(3):
            for p in range(3):
                state = (v, p)
                q_values = np.zeros(self.n_actions)
                
                if v == 0:  # Low traffic
                    q_values[0] = 1.0  # Short green is optimal
                elif v == 1:  # Medium traffic
                    q_values[1] = 1.0  # Medium green is optimal
                else:  # High traffic
                    q_values[2] = 1.0  # Long green is optimal
                
                # Adjust for pedestrians
                if p == 2:  # High pedestrians
                    q_values[3] = 0.8  # Consider emergency mode
                
                self._q_table[state] = q_values
    
    def discretize_state(self, vehicle_count: int, pedestrian_count: int) -> Tuple[int, int]:
        """
        Discretize raw counts into state representation.
        
        Args:
            vehicle_count: Raw vehicle count.
            pedestrian_count: Raw pedestrian count.
            
        Returns:
            Discretized state tuple (vehicle_level, pedestrian_level).
        """
        # Vehicle level
        if vehicle_count < 10:
            v_level = 0  # Low
        elif vehicle_count < 25:
            v_level = 1  # Medium
        else:
            v_level = 2  # High
        
        # Pedestrian level
        if pedestrian_count < 3:
            p_level = 0  # Low
        elif pedestrian_count < 8:
            p_level = 1  # Medium
        else:
            p_level = 2  # High
        
        return (v_level, p_level)
    
    def get_action(self, state: Tuple[int, int] = None, 
                   vehicle_count: int = None, 
                   pedestrian_count: int = None) -> ActionResult:
        """
        Get the optimal action for a given state.
        
        Args:
            state: Discretized state tuple (vehicle_level, pedestrian_level).
            vehicle_count: Raw vehicle count (alternative to state).
            pedestrian_count: Raw pedestrian count (alternative to state).
            
        Returns:
            ActionResult with recommended action.
        """
        self._load_policy()
        
        # Discretize if raw counts provided
        if state is None:
            if vehicle_count is None or pedestrian_count is None:
                raise ValueError("Must provide either state or both vehicle_count and pedestrian_count")
            state = self.discretize_state(vehicle_count, pedestrian_count)
        
        # Handle defaultdict or regular dict
        if isinstance(self._q_table, defaultdict):
            q_values = self._q_table[state]
        else:
            q_values = self._q_table.get(state, np.zeros(self.n_actions))
        
        # Select best action
        action_id = int(np.argmax(q_values))
        expected_reward = float(q_values[action_id])
        
        return ActionResult(
            action_id=action_id,
            action_name=self.ACTION_NAMES[action_id],
            expected_reward=expected_reward,
            state=state,
            q_values={i: float(v) for i, v in enumerate(q_values)}
        )
    
    def get_action_batch(self, states: List[Tuple[int, int]]) -> List[ActionResult]:
        """
        Get optimal actions for a batch of states.
        
        Args:
            states: List of discretized state tuples.
            
        Returns:
            List of ActionResults.
        """
        return [self.get_action(state=s) for s in states]
    
    def get_action_from_features(self, features: np.ndarray) -> List[ActionResult]:
        """
        Get actions from raw feature matrix.
        
        Args:
            features: Feature matrix of shape (n_samples, 2) with
                     [vehicle_count, pedestrian_count] columns.
                     
        Returns:
            List of ActionResults.
        """
        results = []
        for v, p in features:
            result = self.get_action(
                vehicle_count=int(v),
                pedestrian_count=int(p)
            )
            results.append(result)
        return results
    
    def get_policy_info(self) -> dict:
        """
        Get information about the loaded policy.
        
        Returns:
            Dictionary with policy statistics.
        """
        self._load_policy()
        
        return {
            'n_actions': self.n_actions,
            'action_names': self.ACTION_NAMES,
            'n_states': len(self._q_table) if self._q_table else 0,
            'policy_path': str(self._policy_path) if self._policy_path else 'default'
        }
    
    def get_policy_summary(self) -> str:
        """Get a human-readable policy summary."""
        self._load_policy()
        
        lines = ["Traffic Signal Policy Summary", "=" * 40]
        
        v_names = ['Low', 'Medium', 'High']
        p_names = ['Low', 'Medium', 'High']
        
        for state in sorted(self._q_table.keys()):
            q_values = self._q_table[state]
            best_action = int(np.argmax(q_values))
            
            # Handle different state formats
            try:
                if isinstance(state, tuple) and len(state) >= 2:
                    v_idx = min(int(state[0]), 2)
                    p_idx = min(int(state[1]), 2)
                    state_desc = f"Vehicles: {v_names[v_idx]}, Pedestrians: {p_names[p_idx]}"
                else:
                    state_desc = f"State: {state}"
                
                action_desc = self.ACTION_NAMES[best_action] if best_action < len(self.ACTION_NAMES) else f"Action {best_action}"
                lines.append(f"{state_desc} -> {action_desc}")
            except (IndexError, TypeError):
                # Skip malformed states
                continue
        
        return "\n".join(lines)
    
    def save_policy(self, filepath: str):
        """Save the current policy to file."""
        if self._q_table:
            with open(filepath, 'wb') as f:
                pickle.dump(dict(self._q_table), f)
    
    @property
    def is_loaded(self) -> bool:
        """Check if policy is loaded."""
        return self._q_table is not None
