# Robot models

`franka_fr3/` is an unmodified copy of the Franka Research 3 model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3),
taken at commit `0059d4335f8156206f63a35662313385f7ad6d74` (2026-10-07). It is released
under the Apache-2.0 licence included in that folder.

Leave those files as they are. Anything we add (a table, a block, a pusher) goes in
separate files that include `franka_fr3/fr3.xml`, so the robot can be updated or swapped
without touching our work.
