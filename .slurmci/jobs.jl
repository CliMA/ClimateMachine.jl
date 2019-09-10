include("test_set.jl")
[SlurmJob(`.slurmci/cpu-init.sh`) =>
 [
   SlurmJob(`.slurmci/cpu-test.sh`)
   [SlurmJob(`.slurmci/cpu.sh $test`, ntasks=ntasks) for (ntasks, test) in cpu_tests]
   [SlurmJob(`.slurmci/cpu.sh $test`, ntasks=ntasks) for (ntasks, test) in cpu_gpu_tests]
 ],
 SlurmJob(`.slurmci/gpu-init.sh`) =>
 [
   SlurmJob(`.slurmci/gpu-test.sh`)
   [SlurmJob(`.slurmci/gpu.sh $test`, ntasks=ntasks) for (ntasks, test) in gpu_tests]
   [SlurmJob(`.slurmci/gpu.sh $test`, ntasks=ntasks) for (ntasks, test) in cpu_gpu_tests]
 ]
]
