function render_tutorial_step(path, 
                              config="cpu",
                              queue="central",
                              ntasks=1,
                              mem=6000)
    @assert config in ("cpu", "gpu")
    return """
  - label: "$(string(name))"
    command: |
      julia --project=docs/ docs/generate_tutorial.jl "$(string(path))"
      agents:
        config: $(config)
        queue: $(queue)
        slurm_ntasks: $(ntasks)
        slurm_mem_per_cpu: $(mem)
"""
end


function render_buildkite_yaml()
    PREAMPLE = """
env:
  JULIA_VERSION: "1.5.2"
  GKSwstype: nul
  OPENBLAS_NUM_THREADS: 1
  CLIMATEMACHINE_SETTINGS_DISABLE_GPU: "true"
  CLIMATEMACHINE_SETTINGS_FIX_RNG_SEED: "true"
  CLIMATEMACHINE_SETTINGS_DISABLE_CUSTOM_LOGGER: "true"

steps:
  - label: "Build project"
    command:
      - "julia --project --color=yes -e 'using Pkg; Pkg.instantiate()'"
      - "julia --project=docs/ --color=yes -e 'using Pkg; Pkg.instantiate()'"
      - "julia --project=docs/ --color=yes -e 'using Pkg; Pkg.precompile()'"
    agents:
      config: cpu
      queue: central
      slurm_ntasks: 1
      slurm_cpus_per_task: 1
      slurm_mem_per_cpu: 6000

  - wait
"""

end