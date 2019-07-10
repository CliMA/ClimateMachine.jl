MPI.Initialized() || MPI.Init()
Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
mpicomm = MPI.COMM_WORLD
ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
loglevel = ll == "DEBUG" ? Logging.Debug :
ll == "WARN"  ? Logging.Warn  :
ll == "ERROR" ? Logging.Error : Logging.Info
logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
global_logger(ConsoleLogger(logger_stream, loglevel))
@static if haspkg("CUDAnative")
  device!(MPI.Comm_rank(mpicomm) % length(devices()))
end
