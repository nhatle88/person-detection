# python
import tensorrt as trt


def build_engine(onnx_model_path: str, engine_path: str = None, max_batch_size: int = 1,
                 max_workspace_size: int = 1 << 28) -> trt.ICudaEngine:
    """
    Build a TensorRT engine from an ONNX model.

    Parameters:
      onnx_model_path: Path to the ONNX model.
      engine_path: Optional path to save the serialized engine.
      max_batch_size: Maximum batch size.
      max_workspace_size: Maximum GPU workspace size in bytes.

    Returns:
      A TensorRT ICudaEngine.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            error_messages = "\n".join(str(e) for e in errors)
            raise RuntimeError(f"Failed to parse ONNX model:\n{error_messages}")

    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace_size
    engine = builder.build_cuda_engine(network)

    if engine_path:
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())

    return engine