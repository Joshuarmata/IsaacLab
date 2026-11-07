* Added the manager-based ``Isaac-HandleGrab`` and ``Isaac-HandleGrab-Camera`` tasks
  (``isaaclab_tasks.core.official_handle_grab``): the SO-101 handle-grab scene trained the way the NVIDIA SO-101 vial task
  is. They use relative joint-step actions, dense approach shaping with latched grasp and lift milestones, a success
  bonus for holding the strap with the bag lifted, resets from a physics-validated four-phase dataset, an RSL-RL PPO state
  teacher, and a distilled overhead-camera student. ``presets=sim2real`` adds physical randomization and
  ``presets=pinch_only`` makes a held strap enough for success. The direct ``Isaac-HandleGrab-Direct`` task is unchanged.
