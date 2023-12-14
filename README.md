# renewableopt

This project implements a optimization program which designs an economically efficient
100% renewable energy grid. The generation resources in a renewable grid are
intermittent which implies the need for energy storage (ES). The ability for a system design to balance anticipated load is a function of:
1. the "capacity" of generation and ES resources, and
2. the existence of a ES "dispatch" which bridges periods of generation intermittency.

Capacity and dispatch are interdependent: the capacity of ES determines the constraints
on its dispatch while the existence of a dispatch determines the viability of its design.
Therefore, an optimization problem must be posed as a function of both quantities.

Conditions for linearity of the (still informally) proposed problem are provided.

## Disclaimer

This repository does not purport to be a library with an externally-facing interface.
It exists for now as a single-purpose application. The project's future direction is (as of Dec 2023) being deliberated.

## Request for Mathematical Details

A core linear program (LP), implemented in the [SinglePeriodModel](https://github.com/seanjennings960/renewableopt/blob/master/src/renewableopt/optimal_design/single_period.py) class can be viewed for a flavor of the algorithm.

A formal mathematical problem proposition and algorithmic proof exist in an
unpolished (physical) notebook, but can be digitized upon request. The author can be reached via email: sean.jennings1@gmail.com.

## License

`renewableopt` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
