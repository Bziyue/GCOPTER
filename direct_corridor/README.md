# GCOPTER adapters for SweepCorridor

The generator implementation has moved to [SweepCorridor](../SweepCorridor).
It is a C++17/Eigen header-only package with `Generator`, `Corridor` and
shared query interfaces. Copy that directory to integrate into another project.

This directory contains ROS and legacy GCOPTER tool adapters only. The
`gcopter::direct_corridor` CMake target is an INTERFACE forwarding target; it
contains no second generator implementation. New code should link
`SweepCorridor::SweepCorridor` and include `<SweepCorridor.hpp>` directly.

The RViz demo renders native cubic controls and uses C3 exclusively.
Historical experiment sources were removed from active builds; recover them
from commit `92da988`. Their validation data remains in `docs/validation/`.
