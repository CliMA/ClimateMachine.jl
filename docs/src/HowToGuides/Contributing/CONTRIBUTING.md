# Contributing

ClimateMachine encourages Pull Requests (PRs) and contributions.
The easiest way to contribute is by running code and letting us know what's wrong by creating issues.
In the case that you can specifically pinpoint the issue within the code, please consider submitting a PR with these changes.

To contribute, we assume that you already have a basic understanding of git and version control as all tools are accessible on Github; however, we will discuss the general workflow for submitting code for review to ClimateMachine.

If you are unfamiliar with git and version control, please consider reading the following guides:

- [Atlassian (bitbucket) git tutorials. A set of tips and tricks for getting started with git](https://www.atlassian.com/git/tutorials)
- [GitHub's git tutorials. A set of resources from GitHub to learn git](https://try.github.io/)

There are a bunch of other guides available as well.
We will try to guide you along the process here, but this is by no means an exhaustive set of everything you will need to know when using git and version control.

## Forks and branches

To start contributing, first create your own fork (version of ClimateMachine) [on GitHub](https://github.com/CliMA/ClimateMachine.jl) and check out your copy:

```
$ git clone https://github.com/<username>/ClimateMachine.jl.git
$ cd ClimateMachine.jl
$ git remote add upstream https://github.com/CliMA/ClimateMachine.jl.git
```

This will create two places for you to keep code, one called `origin`, which is your own fork and another called `upstream`, which is ClimateMachine's main repository.

From there, it is important to create a *feature branch* for your project:

```
$ git checkout -b <branchname>
```
Creating a feature branch allows you to more easily reconcile your code with the master branch on ClimateMachine.

### Basic git interactions

Firstly, make sure git knows your name and email address:

```
$ git config --global user.name "A. Climate Developer"
$ git config --global user.email "j.climate.developer@eg.com"
```

From there, we need to begin saving different versions of code.
This is done by creating `commits`, which are save states for all code within the git repository.
The basic workflow for creating changes to ClimateMachine is the following:

1. `git status` will show you any changes in your current code from the last commit
2. `git add <FILE>` will add any files you want to commit into a staging area
3. `git commit` will store any staged files into a commit.
4. `git push origin <branchname>` will then push all the changes to `origin`, which is a nickname for your fork on GitHub.
5. When you are happy with all the code on your branch, go to GitHub and click the button to create a pull request with all the changes against ClimateMachine's master branch.

### Squash and Rebase

Before merging with ClimateMachine, use `git rebase` (not `git merge`) to sync your work  with the current master.

```
$ git fetch upstream
$ git rebase upstream/master
```

Once the PR is ready for review (or in the process of review), be sure to [squash your commits](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request#squash-your-changes).
This will help us keep the commit history clean on the master branch of ClimateMachine.

## Formatting and style

For the most part, we follow the [YASGuide](https://github.com/jrevels/YASGuide) for Julia formatting, with small exceptions covered in the [Coding Conventions](https://CliMA.github.io/ClimateMachine.jl/latest/CodingConventions.html) section of the documentation.

In addition to this, once you are happy with your PR, please apply [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) to all changed files in the repository.

### Formatting utility

A convenience utility is located at `.dev/climaformat.jl` that will format the julia files in the given path. For example, from the top-level ClimateMachine directory
```
julia .dev/climaformat.jl src/ClimateMachine.jl
```
will format `src/ClimateMachine.jl` and
```
julia .dev/climaformat.jl .
```
will format all of ClimateMachine's julia files.

### Formatting githook

A `pre-commit` script that can be placed in `$GIT_DIR/hooks/*` which will prevent commits of incorrectly formatted julia code.  It will also provide commands that can be run to format the code correctly.

One may tell git about the script with (from the top-level directory of a clone of ClimateMachine)
```
ln -s ../../.dev/hooks/pre-commit .git/hooks
```
and then when `git commit` is run an error message will be given for julia files that are staged to be committed that are not formatted according to ClimateMachine's standard.  With this, when I try to commit changes to `src/Arrays/MPIStateArrays.jl` that are not formatted correctly I get the following error message.

```
❯ git commit                                                                                                           │
Activating environment at `~/research/code/ClimateMachine.jl/.dev/Project.toml`                                        │
┌ Error: File src/Arrays/MPIStateArrays.jl needs to be indented with:                                                  │
│     julia /home/lucas/research/code/ClimateMachine.jl/.dev/climaformat.jl /home/lucas/research/code/ClimateMachine.jl│
/src/Arrays/MPIStateArrays.jl                                                                                          │
│ and added to the git index via                                                                                       │
│     git add /home/lucas/research/code/ClimateMachine.jl/src/Arrays/MPIStateArrays.jl                                 │
└ @ Main ~/research/code/ClimateMachine.jl/.git/hooks/pre-commit:30
```

See `man 5 githooks` for more information about git hooks.

### Precompiling the JuliaFormatter

To speed up the formatter and the githook, a custom system image can be built with the [`PackageCompiler`]. That said, the following [drawback] from the `PackageCompiler` repository should be noted:

> It should be clearly stated that there are some drawbacks to using a custom
> sysimage, thereby sidestepping the standard Julia package precompilation
> system. The biggest drawback is that packages that are compiled into a
> sysimage (including their dependencies!) are "locked" to the version they
> where at when the sysimage was created. This means that no matter what package
> version you have installed in your current project, the one in the sysimage
> will take precedence. This can lead to bugs where you start with a project
> that needs a specific version of a package, but you have another one compiled
> into the sysimage.

The `PackageCompiler` compiler can be used with the `JuliaFormatter` using the following commands (from a top-level directory of a clone of ClimateMachine)
```
$ julia -q
julia> using Pkg
julia> Pkg.add("PackageCompiler")
julia> using PackageCompiler
julia> Pkg.activate(joinpath(@__DIR__,".dev"))
julia> using PackageCompiler
julia> PackageCompiler.create_sysimage(:JuliaFormatter; precompile_execution_file=joinpath(@__DIR__, ".dev/precompile.jl"), replace_default=true)
```

If you cannot (or do want to) modify the default system image, instead the following commands can be used
```
$ julia -q
julia> using Pkg
julia> Pkg.add("PackageCompiler")
julia> using PackageCompiler
julia> Pkg.activate(joinpath(@__DIR__,".dev"))
julia> PackageCompiler.create_sysimage(:JuliaFormatter; precompile_execution_file=joinpath(@__DIR__, ".dev/precompile.jl"), sysimage_path=joinpath(@__DIR__, ".git/hooks/JuliaFormatterSysimage.so"))
```
In this case hook `pre-commit.sysimage` should be used. That is, one can use the following linking command (from the top-level directory of a clone of ClimateMachine)
```
ln -s ../../.dev/hooks/pre-commit.sysimage .git/hooks/pre-commit
```
Note: By putting the system image in `.git/hooks` it will be protected from calls to `git clean -x`

## Bors and CI

All commits that end up in the ClimateMachine repository must pass Continuous Integration (CI).
When a PR is updated, it will automatically be run on Microsoft Azure for Linux, Windows, and MacOS; however, because ClimateMachine is heterogeneous and must also run on GPU hardware, we also manually launch CI with Bors.

To test to see if all bors CI will pass, please type `bors try` in a separate comment in the PR.
After this, if you are a collaborator you can merge the commit with `bors merge`.
If you are a collaborator and want to test and merge in the same step, use `bors r+`.

### Tests

Most PRs should include tests and these will be reviewed as part of the code review process on GitHub.
Add your tests in the `test/` directory.

[`PackageCompiler`]: https://github.com/JuliaLang/PackageCompiler.jl
[drawback]: https://julialang.github.io/PackageCompiler.jl/dev/sysimages/#Drawbacks-to-custom-sysimages-1
