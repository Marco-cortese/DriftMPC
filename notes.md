# Notes
Random notes on the project, might be useful.



## Acados installation [@matgrand] 
- Warning: This is not tested
- To install I used this command: `cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DACADOS_WITH_QPOASES=OFF ..`
- qpoases gives an error in compilation (on Arch Linux) -> set to `OFF`
- `CMAKE_POLICY_VERSION_MINIMUM=3.5` because I have a very new cmake, and some packages are too old,
  this is a safe workaround