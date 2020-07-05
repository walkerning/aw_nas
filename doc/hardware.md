## Hardware related: Hardware profiling and parsing

![hardware profiling & parsing flow](pics/hardware_profiling.pdf)

TODO: describe the general flow and interface files

`aw_nas` provide a command-line interface `awnas-hw` to orchestrate the hardware-related objective (e.g., latency, energy, etc.) profiling and parsing flow. A complete workflow example is illustrated as follows.

TODO: a complete step-by-step example

`BaseHardwareObjectiveModel`

### Implement the interface for new search spaces

We provide a mixin class `ProfilingSearchSpace`. This interface has two methods that must be implemented:
* `generate_profiling_primitives`: profiling cfgs => return the profiling primitive list
* `parse_profiling_primitives`: primitive hw-related objective list, profiling/hwobj model cfgs => hwobj model

You might need to implement the hardware-related objective model class for the new search space. You can reuse some codes in `aw_nas/hardware/hw_obj_models`.

### Implement the interface for new hardwares

To implement hardware-specific compilation and parsing process, create a new class inheriting `BaseHardwareCompiler`, implement the `compile` and `hwobj_net_to_primitive` methods. As stated before, you can put your new hardware implementation python file into the `AWNAS_HOME/plugins`, to make it accessible by `aw_nas`.