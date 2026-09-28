
"""Flask inference backend. Raw uploads stay in memory and are not written to disk."""
from pathlib import Path
from functools import lru_cache
import io,json,threading,hashlib,base64
from PIL import Image
import numpy as np
import torch
from flask import Flask,request,jsonify
from pneumonia_core import make_model,inference_image,tensor_transform,calibrated_probability

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=21*1024*1024
INFERENCE_LOCK=threading.Lock()

@lru_cache(maxsize=1)
def load_bundle():
    root=Path(__file__).parent/'model'
    meta=json.loads((root/'metadata.json').read_text())
    with (root/'weights.pt').open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=meta.get('weights_sha256'):raise ValueError('Checkpoint fingerprint mismatch.')
    model=make_model(meta['model_name'],pretrained=False).cpu()
    model.load_state_dict(torch.load(root/'weights.pt',map_location='cpu',weights_only=True))
    model.eval()
    return model,meta

@app.get('/health')
def health():
    try:
        _,meta=load_bundle()
        return jsonify(status='ready',model=meta['model_name'],run_mode=meta['run_mode'])
    except Exception:
        return jsonify(error='Model bundle unavailable or invalid.'),503

@app.post('/predict')
def predict_upload():
    upload=request.files.get('file')
    if upload is None:return jsonify(error='Attach one file using form field file.'),400
    extension=Path(upload.filename or '').suffix.lower()
    if extension not in ('.dcm','.png','.jpg','.jpeg'):
        return jsonify(error='Supported extensions: DCM, PNG, JPG, JPEG.'),400
    raw=upload.read(20*1024*1024+1)
    if not raw or len(raw)>20*1024*1024:return jsonify(error='Image must be nonempty and at most 20 MB.'),400
    try:
        model,meta=load_bundle()
        image=inference_image(io.BytesIO(raw),extension)
        x=tensor_transform(meta['model_name'])(image).unsqueeze(0)
        with INFERENCE_LOCK,torch.inference_mode():logit=float(model(x).reshape(-1)[0])
        probability=float(calibrated_probability(logit,meta['calibration_a'],meta['calibration_b']))
        if not np.isfinite(probability):raise ValueError('Nonfinite score.')
        return jsonify(positive_probability=probability,threshold=meta['threshold'],
            predicted_target=int(probability>=meta['threshold']),model=meta['model_name'],
            run_mode=meta['run_mode'],input_domain='evaluated DICOM pipeline' if extension=='.dcm' else 'exploratory raster input')
    except Exception:
        return jsonify(error='Unable to process image. Check encoding, decoder support and model files.'),422

@app.post('/explain')
def explain_upload():
    upload=request.files.get('file')
    if upload is None:return jsonify(error='Attach an image.'),400
    extension=Path(upload.filename or '').suffix.lower()
    if extension not in ('.dcm','.png','.jpg','.jpeg'):return jsonify(error='Unsupported extension.'),400
    raw=upload.read(20*1024*1024+1)
    if not raw or len(raw)>20*1024*1024:return jsonify(error='Invalid image size.'),400
    try:
        model,meta=load_bundle();image=inference_image(io.BytesIO(raw),extension)
        x=tensor_transform(meta['model_name'])(image).unsqueeze(0).requires_grad_(True)
        layers=[m for m in model.modules() if isinstance(m,torch.nn.Conv2d)]
        layer=model[15] if meta['model_name']=='scratch_se_cnn' else layers[-1]
        captured={}
        def capture(module,args,output):captured['activation']=output
        with INFERENCE_LOCK,torch.enable_grad():
            hook=layer.register_forward_hook(capture)
            try:
                score=model(x).reshape(-1)[0];act=captured['activation']
                grad=torch.autograd.grad(score,act)[0]
                cam=torch.relu((act*grad.mean((2,3),keepdim=True)).sum(1,keepdim=True))
                cam=torch.nn.functional.interpolate(cam,size=(224,224),mode='bilinear',align_corners=False)[0,0].detach().numpy()
            finally:hook.remove()
        blank=bool(cam.max()<=0)
        if not blank:cam=cam/cam.max()
        gray=np.asarray(image,dtype=float)/255
        heat=np.stack([cam,np.zeros_like(cam),1-cam],axis=-1)
        overlay=np.repeat(gray[...,None],3,axis=-1) if blank else .65*gray[...,None]+.35*heat
        buf=io.BytesIO();Image.fromarray(np.uint8(np.clip(overlay,0,1)*255)).save(buf,format='PNG')
        return jsonify(png_base64=base64.b64encode(buf.getvalue()).decode('ascii'),blank_map=blank,
                       explanation='Positive-score Grad-CAM; red higher, blue lower. Not lesion localization.')
    except Exception:
        return jsonify(error='Unable to generate attribution.'),422

