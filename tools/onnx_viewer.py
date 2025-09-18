import os
import numpy as np
from PIL import Image
import streamlit as st
import onnxruntime as ort


CITYSCAPE_COLORMAP = {
    0:(128, 64,128), 1:(244, 35,232), 2:( 70, 70, 70), 3:(102,102,156),
    4:(190,153,153), 5:(153,153,153), 6:(250,170, 30), 7:(220,220,  0),
    8:(107,142, 35), 9:(152,251,152), 10:( 70,130,180), 11:(220, 20, 60),
    12:(255,  0,  0), 13:(  0,  0,142), 14:(  0,  0, 70), 15:(  0, 60,100),
    16:(  0, 80,100), 17:(  0,  0,230), 18:(119, 11, 32),
}


def load_onnx_model(model_bytes):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_bytes, providers=providers)
    return session


def preprocess(image, input_size=(256, 256)):
    assert isinstance(image, Image.Image)
    image = image.convert("RGB")
    if image.height != input_size[0] or image.width != input_size[1]:
        image = image.resize(input_size)

    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    normalized = (image - mean) / std
    tensor = normalized.transpose(2, 0, 1)  # HWC -> CHW
    tensor = np.expand_dims(tensor, axis=0) # CHW -> NCHW

    return tensor.astype(np.float32)


def postprocess(pred, colormap=None):
    assert isinstance(pred, np.ndarray)
    if colormap is None:
        colormap = CITYSCAPE_COLORMAP
    else:
        assert isinstance(colormap, (list, tuple, dict))

    pred = pred.squeeze(0)    # NCHW -> CHW
    if pred.ndim == 3:  # Multi-class / Softmax
        pred_mask = np.argmax(pred, axis=0).astype(np.uint8)
    else:  # Binary / Sigmoid
        pred_mask = (pred > 0.5).astype(np.uint8)

    if colormap:
        vis_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        if isinstance(colormap, (list, tuple)):
            colormap = enumerate(colormap)
        else:
            colormap = colormap.items()

        for label, color in colormap:
            vis_mask[pred_mask == label] = color
    else:
        vis_mask = pred_mask * 255

    return Image.fromarray(vis_mask).convert('RGB'), Image.fromarray(pred_mask)


def overlay_segmentation_mask(image, mask, alpha=0.4):
    assert isinstance(image, Image.Image) and isinstance(mask, Image.Image)

    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    overlay = Image.blend(image, mask, alpha=alpha)

    return overlay


def streamlit_ui():
    st.title("Segmentation Viewer")
    
    onnx_file = st.file_uploader("Select an ONNX model", type=["onnx"], key="onnx_model")
    if not onnx_file:
        st.warning("Please upload an ONNX model to continue.")
        return

    try:
        session = load_onnx_model(onnx_file.read())
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        return

    input_shape = session.get_inputs()[0].shape[2:]
    if isinstance(input_shape[0], str) or isinstance(input_shape[1], str):
        st.error("Dynamic input shapes not supported — please use a fixed-shape ONNX model.")
        return

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if st.button("🧹 Clear uploaded images"):
        st.session_state.uploaded_files = []
        st.session_state.uploader_key += 1
        st.rerun()

    st.session_state.uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                                        key=f'uploader_{st.session_state.uploader_key}')
    uploaded_files = st.session_state.uploaded_files

    if uploaded_files:
        st.markdown(f"**🧾 {len(uploaded_files)} file(s) uploaded**")
        st.markdown("### Save Options")

        save_overlay = st.checkbox("💾 Save segmentation overlay")
        save_raw_mask = st.checkbox("💾 Save raw prediction mask")

        save_dir = None
        if save_overlay or save_raw_mask:
            save_dir = st.text_input("📁 Output folder (type or paste path):", value="output")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            clean_output = st.checkbox("🧹 Clean output folder", value=False)
            if clean_output:
                for f in os.listdir(save_dir):
                    try:
                        os.remove(os.path.join(save_dir, f))
                    except Exception as e:
                        st.warning(f"Could not delete {f}: {e}")

        if st.button("▶️ Run Segmentation"):
            if len(uploaded_files) > 1 and not (save_overlay or save_raw_mask):
                st.warning("'Save segmentation overlay' or 'Save raw prediction mask' must be enabled when uploading multiple images.")
                return

            progress = st.progress(0.0)
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                imgname = os.path.basename(uploaded_file.name).split('.')[0]
                if len(uploaded_files) == 1:
                    st.image(image, caption="Original Image", use_container_width=True)

                input_tensor = preprocess(image, input_shape).astype(np.float32)
                ort_inputs = {session.get_inputs()[0].name: input_tensor}
                preds = session.run(None, ort_inputs)
                vis_mask, pred_mask = postprocess(preds[0])

                if len(uploaded_files) == 1 or save_overlay:
                    overlay = overlay_segmentation_mask(image, vis_mask)

                if len(uploaded_files) == 1:
                    st.image(overlay, caption="Segmentation Overlay", use_container_width=True)

                if save_overlay:
                    overlay_path = os.path.join(save_dir, f'{imgname}_overlay.png')
                    overlay.save(overlay_path)

                if save_raw_mask:
                    mask_path = os.path.join(save_dir, f'{imgname}.png')
                    pred_mask.save(mask_path)

                progress.progress((idx+1) / len(uploaded_files))

            st.success('Segmentation completed.')


if __name__ == '__main__':

    streamlit_ui()