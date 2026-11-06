"""Face swap: replace the main face in a video with a source face, keep everything else.

Usage: python3 swap.py SOURCE_IMG INPUT_VIDEO OUTPUT_VIDEO MODELS_DIR [--preview N]
"""
import sys, subprocess, json
import numpy as np
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

src_path, in_path, out_path, models = sys.argv[1:5]
preview = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[5] == "--preview" else None

so = ort.SessionOptions(); so.intra_op_num_threads = 4
app = FaceAnalysis(name="buffalo_l", root=models, allowed_modules=["detection", "recognition"],
                   providers=["CPUExecutionProvider"])
app.models  # loaded from models/models/buffalo_l
app.prepare(ctx_id=-1, det_size=(640, 640))
swapper = get_model(f"{models}/inswapper_128.onnx", providers=["CPUExecutionProvider"])
gfpgan = ort.InferenceSession(f"{models}/gfpgan_1.4.onnx", so, providers=["CPUExecutionProvider"])
g_in = gfpgan.get_inputs()[0].name

src_img = cv2.imread(src_path)
src_faces = app.get(src_img)
assert src_faces, "no face in source image"
src_face = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

# FFHQ 5-point template for 512x512
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)

def soft_mask(size, pad=0.1, blur=0.08):
    m = np.zeros((size, size), np.float32)
    p = int(size * pad)
    m[p:size-p, p:size-p] = 1
    k = int(size * blur) * 2 + 1
    return cv2.GaussianBlur(m, (k, k), 0)

MASK512 = soft_mask(512, pad=0.08, blur=0.07)

def enhance(frame, kps, strength=0.75):
    M = cv2.estimateAffinePartial2D(kps.astype(np.float32), TEMPLATE, method=cv2.LMEDS)[0]
    crop = cv2.warpAffine(frame, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
    x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    y = gfpgan.run(None, {g_in: x})[0][0]
    y = np.clip((y.transpose(1, 2, 0) + 1) / 2, 0, 1)
    y = cv2.cvtColor((y * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    # keep original colour/lighting: match mean/std per channel in LAB
    yl = cv2.cvtColor(y, cv2.COLOR_BGR2LAB).astype(np.float32)
    cl = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    w = MASK512[..., None] > 0.5
    for c in range(3):
        ys, cs = yl[..., c][w[..., 0]], cl[..., c][w[..., 0]]
        yl[..., c] = (yl[..., c] - ys.mean()) / (ys.std() + 1e-6) * cs.std() + cs.mean()
    y = cv2.cvtColor(np.clip(yl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    y = cv2.addWeighted(y, strength, crop, 1 - strength, 0)
    # add back a touch of the frame's natural grain so the face isn't plastic-smooth
    h, w_ = frame.shape[:2]
    IM = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(y, IM, (w_, h), flags=cv2.INTER_LANCZOS4)
    mask = cv2.warpAffine(MASK512, IM, (w_, h))[..., None]
    return (back * mask + frame * (1 - mask)).astype(np.uint8)

def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "json", path])
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]

W, H = probe(in_path)
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", in_path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       stdout=subprocess.PIPE)
if preview is None:
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-i", in_path,
                            "-map", "0:v", "-map", "1:a", "-c:a", "copy",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                            "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709",
                            "-color_primaries", "bt709", "-movflags", "+faststart", "-shortest", out_path],
                           stdin=subprocess.PIPE)

fsize = W * H * 3
i = 0
last_kps = None
while True:
    buf = dec.stdout.read(fsize)
    if len(buf) < fsize:
        break
    frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    if preview is None or i % preview == 0:
        faces = app.get(frame)
        if faces:
            tgt = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            out = swapper.get(frame, tgt, src_face, paste_back=True)
            out = enhance(out, tgt.kps)
        else:
            out = frame
        if preview is not None:
            cv2.imwrite(f"{out_path}_{i:04d}.jpg", np.hstack([frame, out]))
    else:
        out = frame
    if preview is None:
        enc.stdin.write(out.tobytes())
    i += 1
    if i % 30 == 0:
        print(f"frame {i}", flush=True)

dec.wait()
if preview is None:
    enc.stdin.close(); enc.wait()
print("done", i)
