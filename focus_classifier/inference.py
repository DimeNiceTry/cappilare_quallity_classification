"""In-memory inference for uploads using independently saved final models."""
import csv
import hashlib
import io
import json
import threading
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from .features import BASIC, from_gray
from .modeling import predict_bundle

APP=Path(__file__).resolve().parent
MAX_UPLOAD=20*1024*1024
MAX_PIXELS=16_000_000
MODEL_ORDER=['neural','trees_extended','logistic_extended','logistic_basic','trees_basic',
             'single_laplacian_raw','single_laplacian_smooth','single_tenengrad_smooth','single_local_tenengrad_p75']
TITLES={'neural':'MobileNetV3 · по изображению','trees_extended':'ExtraTrees · 43 признака',
        'logistic_extended':'Линейная модель · 43 признака','logistic_basic':'Линейная модель · 4 признака',
        'trees_basic':'ExtraTrees · 4 признака','single_laplacian_raw':'Только лапласиан',
        'single_laplacian_smooth':'Лапласиан после сглаживания','single_tenengrad_smooth':'Tenengrad всего кадра',
        'single_local_tenengrad_p75':'Локальный Tenengrad p75'}


def decode_upload(raw,resize=False):
    if not raw or len(raw)>MAX_UPLOAD:raise ValueError('Размер файла должен быть от 1 байта до 20 МБ.')
    try:
        with Image.open(io.BytesIO(raw)) as im:
            width,height=im.size;fmt=im.format;mode=im.mode
            if width*height>MAX_PIXELS:raise ValueError('Изображение слишком большое: максимум 16 миллионов пикселей.')
            if fmt not in ('PNG','JPEG','BMP','TIFF'):raise ValueError('Поддерживаются PNG, JPEG, BMP и TIFF.')
            if mode.startswith('I') or mode=='F':raise ValueError('Нужна 8-битная фотография. 16-битные изображения сначала преобразуй в 8 бит.')
            if getattr(im,'n_frames',1)!=1:raise ValueError('Загрузи отдельный кадр, а не многостраничное изображение.')
            im.verify()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
        raise ValueError('Файл не удалось прочитать как фотографию.') from error
    native=(width,height)==(1024,768)
    if not native and not resize:
        raise ValueError(f'Размер {width}×{height}. Модели проверены на 1024×768. Для пробной оценки включи преобразование размера.')
    if not native and abs(width/height-4/3)>.02:
        raise ValueError('Для преобразования нужен кадр с пропорциями 4:3. Изображение не растягивается и не обрезается автоматически.')
    encoded=np.frombuffer(raw,np.uint8)
    gray=cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE)
    bgr=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if gray is None or bgr is None:raise ValueError('Не удалось декодировать изображение.')
    if not native:
        interpolation=cv2.INTER_AREA if width>1024 else cv2.INTER_LINEAR
        gray=cv2.resize(gray,(1024,768),interpolation=interpolation)
        bgr=cv2.resize(bgr,(1024,768),interpolation=interpolation)
    return gray,bgr,dict(width=width,height=height,format=fmt,resized=not native,evaluated_width=1024,evaluated_height=768)


class InferenceEngine:
    def __init__(self):
        self.lock=threading.RLock();self.models=None;self.backbone=None
        self.metrics=json.loads((APP/'artifacts/metrics.json').read_text(encoding='utf-8'))
        gallery=json.loads((APP/'artifacts/gallery.json').read_text(encoding='utf-8'))
        truth={r['file']:r['truth'] for r in gallery['rows']}
        with (APP/'artifacts/features.csv').open(encoding='utf-8') as h:
            self.known={r['sha256']:dict(file=r['file'],in_training=truth[r['file']]>=0) for r in csv.DictReader(h)}

    def load_models(self):
        if self.models is None:
            models=joblib.load(APP/'artifacts/models_features.joblib')
            models.update(joblib.load(APP/'artifacts/models_neural.joblib'))
            self.models=models

    def catalog(self):
        with self.lock:
            self.load_models()
            return [dict(id=k,name=TITLES[k],thresholds=self.models[k]['thresholds'],
                         auroc=self.metrics['models'][k]['calibrated']['auroc'],
                         precise=self.metrics['models'][k]['precise'],balanced=self.metrics['models'][k]['balanced'])
                    for k in MODEL_ORDER]

    def predict(self,raw,model_ids,mode='precise',resize=False,manual_threshold=None):
        if not isinstance(model_ids,list) or not model_ids or len(set(model_ids))!=len(model_ids) or any(k not in MODEL_ORDER for k in model_ids):
            raise ValueError('Выбери хотя бы одну доступную модель без повторений.')
        if mode not in ('balanced','precise','manual'):raise ValueError('Неизвестный режим отбора.')
        if mode=='manual' and (not isinstance(manual_threshold,(float,int)) or not np.isfinite(manual_threshold) or not 0<=manual_threshold<=1):
            raise ValueError('Ручной порог должен быть числом от 0 до 1.')
        start=time.perf_counter()
        gray,bgr,info=decode_upload(raw,resize=resize)
        decoding_ms=1000*(time.perf_counter()-start)
        with self.lock:
            cv2.setNumThreads(1)
            self.load_models()
            cold_neural='neural' in model_ids and self.backbone is None
            loading_start=time.perf_counter()
            if cold_neural:
                # Web inference uses only the local weights; never downloads models.
                if not (APP/'artifacts/mobilenet_v3_small_imagenet.pt').is_file():
                    raise RuntimeError('Локальные веса MobileNet отсутствуют.')
                from .neural import load_backbone
                self.backbone=load_backbone()
            loading_ms=1000*(time.perf_counter()-loading_start)
            feature_ids=[k for k in model_ids if k!='neural']
            feature_sets={};feature_times={}
            for full in (False,True):
                if any(k.endswith('_extended')==full for k in feature_ids):
                    feature_start=time.perf_counter()
                    feature_sets[full]=from_gray(gray,basic_only=not full)
                    feature_times[full]=1000*(time.perf_counter()-feature_start)
            f=feature_sets.get(True,feature_sets.get(False,{}))
            embedding_ms=0.
            if 'neural' in model_ids:
                from .neural import embedding_from_bgr
                neural_start=time.perf_counter();embedding=embedding_from_bgr(self.backbone,bgr).reshape(1,-1)
                embedding_ms=1000*(time.perf_counter()-neural_start)
            results=[]
            for k in model_ids:
                b=self.models[k];head_start=time.perf_counter()
                x=embedding if k=='neural' else np.array([[feature_sets[k.endswith('_extended')][n] for n in b['features']]])
                p,raw_p=predict_bundle(b,x)
                threshold=float(manual_threshold if mode=='manual' else b['thresholds'][mode])
                decision='disabled' if threshold>1 else 'ok' if p[0]>=threshold else 'not_ok'
                head_ms=1000*(time.perf_counter()-head_start)
                results.append(dict(model=k,name=TITLES[k],p_ok=float(p[0]),p_ok_raw=float(raw_p[0]),threshold=threshold,
                                    decision=decision,analysis_ms=(embedding_ms if k=='neural' else feature_times[k.endswith('_extended')])+head_ms))
        digest=hashlib.sha256(raw).hexdigest()
        return dict(image=info,results=results,mode=mode,features={k:float(f[k]) for k in BASIC if k in f},
                    known_image=self.known.get(digest),timing=dict(decode_ms=decoding_ms,model_loading_ms=loading_ms,
                    cold_neural=cold_neural,total_ms=1000*(time.perf_counter()-start)),
                    note='Финальные модели обучены на 565 согласованных кадрах. Вероятность относится к текущему критерию пригодности.')
