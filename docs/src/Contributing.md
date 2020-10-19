# Contributing

Thank you for considering contributing to the `ClimateMachine`! We encourage
Pull Requests (PRs). Please do not hesitate to ask as questions if you're
unsure about how to help.

## What to contribute?

- The easiest way to contribute is by running the `ClimateMachine`, identifying
  problems and opening issues.

- You can tackle an existing issue. We have a list of good [first
  issues](https://github.com/CliMA/ClimateMachine.jl/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

- Write an example or tutorial.

- Improve documentation or comments if you found something hard to use.

- Implement a new feature if you need it to use the `ClimateMachine`.

## Using `git`

If you are unfamiliar with `git` and version control, the following guides
will be helpful:

- [Atlassian (bitbucket) `git`
  tutorials](https://www.atlassian.com/git/tutorials). A set of tips and tricks
  for getting started with `git`.
- [GitHub's `git` tutorials](https://try.github.io/). A set of resources from
  GitHub to learn `git`.

We provide a brief guide here.

### Identity

First make sure `git` knows your name and email address:

```
$ git config --global user.name "A. Climate Developer"
$ git config --global user.email "a.climate.developer@eg.com"
```

### Forks and branches

Create your own fork of the `ClimateMachine` [on
GitHub](https://github.com/CliMA/ClimateMachine.jl) and check out your copy:

```
$ git clone https://github.com/<username>/ClimateMachine.jl.git
$ cd ClimateMachine.jl
$ git remote add upstream https://github.com/CliMA/ClimateMachine.jl
```

Now you have two remote repositories -- `origin`, which is your fork of the
`ClimateMachine`, and `upstream`, which is the main `ClimateMachine.jl`
repository.

Create a branch for your feature; this will hold your contribution:

```
$ git checkout -b <branchname>
```

## Develop your feature

Follow the [Coding conventions](@ref) we use. Make sure you add tests
for your code in `test/` and appropriate documentation in the code and/or
in `docs/`.

When your PR is ready for review, clean up your commit history by squashing
and make sure your code is current with `ClimateMachine` master by rebasing.

### Squash and rebase

Use `git rebase` (not `git merge`) to sync your work:

```
$ git fetch upstream
$ git rebase upstream/master
```

You might find it easier to [squash your
commits](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request#squash-your-changes)
first.

## Continuous integration

It's time to click the button to open your PR! Fill out the template and
provide a clear summary of what your PR does. When a PR is created or
updated, a set of automated tests are run on the PR in our continuous
integration (CI) system.

A `ClimateMachine` developer will look at your PR and provide feedback!

### Unit testing

Currently a number of checks are run per commit for a given PR.

- `JuliaFormatter` checks if the PR is formatted with `.dev/climaformat.jl`.
- `Documentation` rebuilds the documentation for the PR and checks if the docs are consistent and generate valid output.
- `Unit Tests` run subsets of the unit tests defined in `tests/`, using `Pkg.test()`.
   The tests are run in parallel to ensure that they finish in a reasonable time.
   The tests only run the latest commit for a PR, branch and will kill any stale jobs on push.
   These tests are only run on linux (Ubuntu LTS).

Unit tests are run against every new commit for a given PR,
the status of the unit-tests are not checked during the merge
process but act as a sanity check for developers and reviewers.
Depending on the content changed in the PR, some CI checks that
are not necessary will be skipped.  For example doc only changes
do not require the unit tests to be run.

### The merge process

We use [`bors`](https://bors.tech/) to manage merging PR's in the the `ClimateMachine` repo.
If you're a collaborator and have the necessary permissions, you can type
`bors try` in a comment on a PR to have integration test suite run on that
PR, or `bors r+` to try and merge the code.  Bors ensures that all integration tests
for a given PR always pass before merging into `master`.

### Integration testing

Currently a number of checks are run during integration testing before being merged into master.

- `JuliaFormatter` checks if the PR is formatted with `.dev/climaformat.jl`.
- `Documentation` checks that the documentation correctly builds for the merged PR.
- `OS Unit Tests` checks that ClimateMachine.jl package unit tests can pass
   on every OS supported with a pre-compiled system image (Linux, macOS, Windows).
- `ClimateMachine-CI` computationally expensive integration testing on CPU and GPU hardware using HPC cluster resources.

Integration tests are run when triggered by a reviewer through `bors`.
Integration tests are more computationally heavyweight than unit-tests and can exercise tests using accelerator hardware (GPUs).

Currently HPC cluster integration tests are run using the [Buildkite CI service](https://buildkite.com/clima/climatemachine-ci).
Tests are parallelized and run as individual [Slurm](https://slurm.schedmd.com/documentation.html)
batch jobs on the HPC cluster and defined in `.buildkite/pipeline.yml`.

An example integration test definition is outlined below:
```
  - label: "gpu_soil_test_bc"
    key: "gpu_soil_test_bc"
    command:
      - "mpiexec julia --color=yes --project test/Land/Model/test_bc.jl "
    agents:
      config: gpu
      queue: central
      slurm_ntasks: 1
      slurm_gres: "gpu:1"
```

* label / key: unique test defintion strings
* command(s): list of one or more bash commands to run.
* agent block:
    - `config`: Defines `cpu` or `gpu` test environments.
    - `queue`: HPC queue to submit the job (default `central`).
    - `slurm_`: All `slurm_` definitions are passed through as
       [slurm batch job cli options](https://slurm.schedmd.com/sbatch.html).
       Ex. for the above the `slurm_ntasks: 1` is eqv. to `--ntasks=1`.
       Flags are defined with an empty value.

## Contributing Documentation

Documentation is written in Julia-flavored markdown and generated from two sources:
```
$CLIMATEMACHINE_HOME/docs/src
```
And [Literate.jl](https://fredrikekre.github.io/Literate.jl/v2/) tutorials:
```
$CLIMATEMACHINE_HOME/tutorials
```

To locally build the documentation you need to create a new `docs` project
to build and install the documentation related dependencies:

```
cd $CLIMATEMACHINE_HOME
julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The makefile script will generate the appropriate markdown files and
static html from both the `docs/src` and `tutorials/` directories,
saving the output in `docs/src/generated`.

### How to generate a literate tutorial file

To create a tutorial using ClimateMachine, please use
[Literate.jl](https://github.com/fredrikekre/Literate.jl),
and consult the [Literate documentation](https://fredrikekre.github.io/Literate.jl/stable/)
for questions. For now, all literate tutorials are held in
the `tutorials` directory.

With Literate, all comments turn into markdown text and any
Julia code is read and run *as if it is in the Julia REPL*.
As a small caveat to this, you might need to suppress the
output of certain commands. For example, if you define and
run the following function

```
function f()
    return x = [i * i for i in 1:10]
end

x = f()
```

The entire list will be output, while

```
f();
```

does not (because of the `;`).

To show plots, you may do something like the following:

```
using Plots
plot(x)
```

Please consider writing the comments in your tutorial as if they are meant to be read as an *article explaining the topic the tutorial is meant to explain.*
If there are any specific nuances to writing Literate documentation for ClimateMachine, please let us know!


### Speeding up the documentation build process
Building the tutorials can take a long time so there is an environment variable switch to toggle on / off building the tutorials (`true` deafult):

```
CLIMATEMACHINE_DOCS_GENERATE_TUTORIALS=false julia --project=docs/ docs/make.jl
```
