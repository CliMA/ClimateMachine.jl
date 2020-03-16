# Contributing

CLIMA encourages Pull Requests (PRs) and contributions.
The easiest way to contribute is by running code and letting us know what's wrong by creating issues.
In the case that you can specifically pinpoint the issue within the code, please consider submitting a PR with these changes.

To contribute, we assume that you already have a basic understanding of git and version control as all tools are accessible on Github; however, we will discuss the general workflow for submitting code for review to CLIMA.

If you are unfamiliar with git and version control, please consider reading the following guides:

- [Atlassian (bitbucket) git tutorials. A set of tips and tricks for getting started with git](https://www.atlassian.com/git/tutorials)
- [Github's git tutorials. A set of resources from github to learn git](https://try.github.io/)

There are a bunch of other guides available as well.
We will try to guide you along the process here, but this is by no means an exhaustive set of everything you will need to know when using git and version control.

## Forks and branches

To start contributing, first create your own fork (version of CLIMA) [on GitHub](https://github.com/climate-machine/CLIMA) and check out your copy:

```
$ git clone https://github.com/<username>/CLIMA.git
$ cd CLIMA
$ git remote add upstream https://github.com/climate-machine/CLIMA.git
```

This will create two places for you to keep code, one called `origin`, which is your own fork and another called `upstream`, which is CLIMA's main repository.

From there, it is important to create a *feature branch* for your project:

```
$ git checkout -b <branchname>
```
Creating a feature branch allows you to more easily reconcile your code with the master branch on CLIMA.

### Basic git interactions

Firstly, make sure git knows your name and email address:

```
$ git config --global user.name "A. Climate Developer"
$ git config --global user.email "j.climate.developer@eg.com"
```

From there, we need to begin saving different versions of code.
This is done by creating `commits`, which are save states for all code within the git repository.
The basic workflow for creating changes to CLIMA is the following:

1. `git status` will show you any changes in your current code from the last commit
2. `git add <FILE>` will add any files you want to commit into a staging area
3. `git commit` will store any staged files into a commit.
4. `git push origin <branchname>` will then push all the changes to `origin`, which is a nickname for your fork on github.
5. When you are happy with all the code on your branch, go to github and click the button to create a pull request with all the changes against CLIMA's master branch.

### Squash and Rebase

Before merging with CLIMA, use `git rebase` (not `git merge`) to sync your work  with the current master.

```
$ git fetch upstream
$ git rebase upstream/master
```

Once the PR is ready for review (or in the process of review), be sure to [squash your commits](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request#squash-your-changes).
This will help us keep the commit history clean on the master branch of CLIMA.

## Formatting and style

For the most part, we follow the [YASGuide](https://github.com/jrevels/YASGuide) for Julia formatting, with small exceptions covered in the [Coding Conventions](https://climate-machine.github.io/CLIMA/latest/CodingConventions.html) section of the documentation.

In addition to this, once you are happy with your PR, please apply [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) to all changed files in the repository.

To apply our formatter settings to all changed files, run:
```
julia .dev/format.jl
```

Formatting changes should be done as a separate commit, ideally the last commit of the pull request (you may want to leave it until all other changes have been approved).

### Formatting utility

A convenience utility is located at `.dev/climaformat.jl` that will format the julia files in the given path. For example, from the top-level CLIMA directory
```
julia .dev/climaformat.jl src/CLIMA.jl
```
will format `src/CLIMA.jl` and
```
julia .dev/climaformat.jl .
```
will format all of CLIMA's julia files.

### Formatting githook

A `pre-commit` script that can be placed in `$GIT_DIR/hooks/*` which will prevent commits of incorrectly formatted julia code.  It will also provide commands that can be run to format the code correctly.

One may tell git about the script with (from the top-level directory of a clone of CLIMA)
```
ln -s ../../.dev/hooks/pre-commit .git/hooks
```
and then when `git commit` is run an error message will be given for julia files that are staged to be committed that are not formatted according to CLIMA's standard.  With this, when I try to commit changes to `src/Arrays/MPIStateArrays.jl` that are not formatted correctly I get the following error message.

```
❯ git commit                                                                                                           │
Activating environment at `~/research/code/CLIMA/.dev/Project.toml`                                                    │
┌ Error: File src/Arrays/MPIStateArrays.jl needs to be indented with:                                                  │
│     julia /home/lucas/research/code/CLIMA/.dev/climaformat.jl /home/lucas/research/code/CLIMA/src/Arrays/MPIStateArra│
ys.jl                                                                                                                  │
│ and added to the git index via                                                                                       │
│     git add /home/lucas/research/code/CLIMA/src/Arrays/MPIStateArrays.jl                                             │
└ @ Main ~/research/code/CLIMA/.git/hooks/pre-commit:30
```

See `man 5 githooks` for more information about git hooks.

## Bors and CI

All commits that end up in the CLIMA repository must pass Continuous Integration (CI).
When a PR is updated, it will automatically be run on Microsoft Azure for Linux, Windows, and MacOS; however, because CLIMA is heterogeneous and must also run on GPU hardware, we also manually launch CI with Bors.

To test to see if all bors CI will pass, please type `bors try` in a separate comment in the PR.
After this, if you are a collaborator you can merge the commit with `bors merge`.
If you are a collaborator and want to test and merge in the same step, use `bors r+`.

### Tests

Most PRs should include tests and these will be reviewed as part of the code review process on github.
Add your tests in the `test/` directory.
