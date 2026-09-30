import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def create_identity_model(filename: str, model_name: str):
    """Creates a 1:1 image reconstruction model: [1, 3, 128, 128] -> [1, 3, 128, 128]"""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 128, 128])

    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], model_name, [X], [Y])
    model = helper.make_model(
        graph,
        producer_name="genai-scaffold",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    filepath = os.path.join(MODELS_DIR, filename)
    onnx.save(model, filepath)
    print(f"Generated {filename} -> {filepath}")


def create_classifier_model(filename: str):
    """Creates a classifier model: [1, 3, 128, 128] -> [1, 4]"""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    pool_out = helper.make_tensor_value_info("pool_out", TensorProto.FLOAT, [1, 3, 1, 1])
    flat_out = helper.make_tensor_value_info("flat_out", TensorProto.FLOAT, [1, 3])
    gemm_out = helper.make_tensor_value_info("gemm_out", TensorProto.FLOAT, [1, 4])

    w_data = np.array(
        [
            [1.0, 1.0, 1.0],  # clean
            [0.5, 0.2, 0.8],  # salt_pepper
            [0.2, 0.7, 0.4],  # blur
            [0.3, 0.4, 0.9],  # occlusion
        ],
        dtype=np.float32,
    )
    b_data = np.array([0.5, 0.1, 0.1, 0.1], dtype=np.float32)

    w_init = numpy_helper.from_array(w_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="B")

    node_pool = helper.make_node("GlobalAveragePool", ["input"], ["pool_out"])
    node_flat = helper.make_node("Flatten", ["pool_out"], ["flat_out"], axis=1)
    node_gemm = helper.make_node(
        "Gemm", ["flat_out", "W", "B"], ["gemm_out"], transB=1
    )
    node_softmax = helper.make_node("Softmax", ["gemm_out"], ["output"], axis=1)

    graph = helper.make_graph(
        [node_pool, node_flat, node_gemm, node_softmax],
        "Task2Classifier",
        [X],
        [Y],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(
        graph,
        producer_name="genai-scaffold",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    filepath = os.path.join(MODELS_DIR, filename)
    onnx.save(model, filepath)
    print(f"Generated {filename} -> {filepath}")


def main():
    print(">> Generating initial baseline ONNX models in:", MODELS_DIR)
    create_identity_model("task1_universal_ae.onnx", "Task1UniversalAE")
    create_classifier_model("task2_classifier.onnx")
    create_identity_model("task2_specialist_salt.onnx", "Task2SpecialistSalt")
    create_identity_model("task2_specialist_blur.onnx", "Task2SpecialistBlur")
    create_identity_model("task2_specialist_occlusion.onnx", "Task2SpecialistOcclusion")
    create_identity_model("task3_soft_moe.onnx", "Task3SoftMoE")
    create_identity_model("task4_generator.onnx", "Task4Generator")

    print(">> Verifying models with onnxruntime...")
    for f in [
        "task1_universal_ae.onnx",
        "task2_classifier.onnx",
        "task2_specialist_salt.onnx",
        "task2_specialist_blur.onnx",
        "task2_specialist_occlusion.onnx",
        "task3_soft_moe.onnx",
        "task4_generator.onnx",
    ]:
        p = os.path.join(MODELS_DIR, f)
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        inp = np.zeros((1, 3, 128, 128), dtype=np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: inp})
        print(f"  Verified {f}: output shape = {[o.shape for o in out]}")
    print(">> All 7 ONNX models generated and verified successfully!")


if __name__ == "__main__":
    main()

