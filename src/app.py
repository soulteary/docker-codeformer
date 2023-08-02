"""
inspired by https://huggingface.co/spaces/sczhou/CodeFormer
"""
import os
import cv2
import torch
import gradio as gr

from torchvision.transforms.functional import normalize

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.misc import gpu_is_available, get_device
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY

from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

model_realesrgan = "weights/realesrgan/RealESRGAN_x2plus.pth"
model_codeformer = "weights/CodeFormer/codeformer.pth"
model_detection = "retinaface_resnet50"
device = get_device()

def set_realesrgan():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(model_path=model_realesrgan, model=model, tile=400, tile_pad=40, pre_pad=0, half=True if gpu_is_available() else False, scale=2)
    return upsampler

def set_codeformer_net():
    model = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).to(device)
    model.load_state_dict(torch.load(model_codeformer)["params_ema"])
    model.eval()
    return model

def set_face_helper(upscale):
    face_helper = FaceRestoreHelper(upscale, face_size=512, crop_ratio=(1, 1), det_model=model_detection, save_ext="png", use_parse=True, device=device)
    return face_helper

def set_upscale(upscale, img):
    upscale = int(upscale)
    if upscale > 4:
        upscale = 4 
    if upscale > 2 and max(img.shape[:2])>1000:
        upscale = 2 
    if max(img.shape[:2]) > 1500 or upscale <= 0:
        upscale = 1
    return upscale

codeformer_net = set_codeformer_net()
upsampler = set_realesrgan()

os.makedirs('output', exist_ok=True)

def inference(image, background_enhance, face_upsample, upscale, codeformer_fidelity):
    try:
        has_aligned = False
        only_center_face = False
        draw_box = False
        print('Inp:', image, background_enhance, face_upsample, upscale, codeformer_fidelity)

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        print('\timage size:', img.shape)

        upscale = set_upscale(upscale, img)
        if upscale == 1:
            background_enhance = False
            face_upsample = False

        face_helper = set_face_helper(upscale)
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            if face_helper.is_gray:
                print('\tgrayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            num_det_faces = face_helper.get_face_landmarks_5(only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            face_helper.align_warp_face()

        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        if not has_aligned:
            if bg_upsampler is not None:
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

        save_path = f'output/out.png'
        imwrite(restored_img, str(save_path))

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img, save_path
    except Exception as error:
        print('Global exception', error)
        return None, None

app = gr.Interface(
    inference, [
        gr.inputs.Image(type="filepath", label="Input"),
        gr.inputs.Checkbox(default=True, label="Background_Enhance"),
        gr.inputs.Checkbox(default=True, label="Face_Upsample"),
        gr.inputs.Number(default=2, label="Rescaling_Factor (up to 4)"),
        gr.Slider(0, 1, value=0.5, step=0.01, label='Codeformer_Fidelity (0 for better quality, 1 for better identity)')
    ], [
        gr.Image(type="numpy", visible=True, elem_id="img-refiner"),
        gr.File(label="Download the output", elem_id="download")
    ],
    title="CodeFormer - 强大的面部恢复和增强网络",
    description="<center><img src='/file=/app/CodeFormer/assets/image/logo.png' alt='CodeFormer logo'></center><p>Official Gradio demo</b> for <a href='https://github.com/sczhou/CodeFormer' target='_blank'><b>Towards Robust Blind Face Restoration with Codebook Lookup Transformer (NeurIPS 2022)</b></a></p><p style='text-align: center'>source code: <a href='https://github.com/soulteary/docker-codeformer' target='_blank'>soulteary/docker-codeformer</a></p>",
    article="<p style='text-align: center'>written by: <a href='https://github.com/soulteary/' target='_blank'>@soulteary</a></p>",
    examples=[
        ['assets/image/01.png', True, True, 2, 0.7],
        ['assets/image/02.jpg', True, True, 2, 0.7],
        ['assets/image/03.jpg', True, True, 2, 0.7],
        ['assets/image/04.jpg', True, True, 2, 0.1],
        ['assets/image/05.jpg', True, True, 2, 0.1]
      ],
    )
app.queue(concurrency_count=2)
app.launch(server_name="0.0.0.0")