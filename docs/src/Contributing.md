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
$ git remote add upstream https://github.com/CliMA/ClimateMachine.jl.git
```

Now you have two remote repositories -- `origin`, which is your fork of the
`ClimateMachine`, and `upstream`, which is the main `ClimateMachine.jl`
repository.

Create a branch for your feature; this will hold your contribution:

```
$ git checkout -b <branchname>
```

## Develop your feature

Follow the [coding style](CodingStyle.md) we use. Make sure you add tests
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

We use [`bors`](https://bors.tech/) to manage the `ClimateMachine` repo.
If you're a collaborator and have the necessary permissions, you can type
`bors try` in a comment on a PR to have some additional tests run on that
PR, or `bors r+` to try and merge the code.
