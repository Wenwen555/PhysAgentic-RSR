# Independently reconstructed Sim assets

`visual_stage3_v1/` is the visual Stage 3 reconstruction with a calibrated robot, **not** the original World. Its original geometry lacks a complete graspable lever; it is preserved as an independent historical reconstruction source.

`task_grounded_v1/` contains three task-local Robot-Space contact models, `robot_m0.xml`, `robot_m1.xml`, and `robot_m2.xml`, with nearby parameter JSON sidecars. These are built from a robot contact demonstration and public robot Probe responses; this is the main M0-R → M2-R chain. See `docs/ROBOT_SPACE_M0_M2_IMPLEMENTATION.md` for the completed steps and what remains unverified.
