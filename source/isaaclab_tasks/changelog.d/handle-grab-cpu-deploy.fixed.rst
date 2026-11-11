* Fixed the handle-grab real-arm deploy loop stopping on a CPU-only machine. The ONNX session
  used one thread with spinning disabled, and the image kernels are compiled before the timed loop, so the
  default runtime pool no longer starves the camera crop past the 30 Hz budget.
