# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.legged_env import LeggedEnv
from legged_lab.envs.base.tt_env import TTEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from legged_lab.envs.gr2.gr2_config import (
    GR2FlatAgentCfg,
    GR2FlatEnvCfg,
    GR2RoughAgentCfg,
    GR2RoughEnvCfg,
)
from legged_lab.envs.h1.h1_config import (
    H1FlatAgentCfg,
    H1FlatEnvCfg,
    H1RoughAgentCfg,
    H1RoughEnvCfg,
)
from legged_lab.envs.digit.digit_config import (
    DigitFlatAgentCfg,
    DigitFlatEnvCfg,
    DigitFlatEnvCfg_Play,
    DigitRoughAgentCfg,
    DigitRoughEnvCfg,
    DigitFlatEnvPhaseCfg
)
from legged_lab.envs.t1.t1_config import (
    T1FlatAgentCfg,
    T1FlatEnvCfg,
    T1RoughAgentCfg,
    T1RoughEnvCfg,
)

from legged_lab.envs.t1_tt.t1_tt_config import (
    T1TableTennisEnvCfg,
    T1TableTennisAgentCfg,
    T1TT_EvalEnvCfg,
    # T1TT_S2_EnvCfg,
    # T1TT_S2_AgentCfg
)


from legged_lab.utils.task_registry import task_registry

task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("gr2_flat", BaseEnv, GR2FlatEnvCfg(), GR2FlatAgentCfg())
task_registry.register("gr2_rough", BaseEnv, GR2RoughEnvCfg(), GR2RoughAgentCfg())
task_registry.register("digit_flat", LeggedEnv, DigitFlatEnvCfg(), DigitFlatAgentCfg())
task_registry.register("digit_flat_play", LeggedEnv, DigitFlatEnvCfg_Play(), DigitFlatAgentCfg())
task_registry.register("digit_rough", LeggedEnv, DigitRoughEnvCfg(), DigitRoughAgentCfg())
task_registry.register("digit_flat_phase", LeggedEnv, DigitFlatEnvPhaseCfg(), DigitFlatAgentCfg())
task_registry.register("t1_flat", BaseEnv, T1FlatEnvCfg(), T1FlatAgentCfg())
task_registry.register("t1_rough", BaseEnv, T1RoughEnvCfg(), T1RoughAgentCfg())
task_registry.register("t1_tt", TTEnv, T1TableTennisEnvCfg(), T1TableTennisAgentCfg()) #TTEnv
task_registry.register("t1_tt_eval", TTEnv, T1TT_EvalEnvCfg(), T1TableTennisAgentCfg()) 
# task_registry.register("t1_tt_s2", TTEnv, T1TT_S2_EnvCfg(), T1TT_S2_AgentCfg()) #TTEnv curriculum stage 2