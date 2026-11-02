import io
import json
import secrets
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
import cv2
import numpy as np
from .inference import InferenceEngine,decode_upload,MAX_UPLOAD,APP
from .modeling import predict_bundle
from .serve import Handler


class QuietHandler(Handler):
    def log_message(self,*args):pass


class UploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=InferenceEngine()
        gallery=json.loads((APP/'artifacts/gallery.json').read_text(encoding='utf-8'))
        cls.file=next(r['file'] for r in gallery['rows'] if r['truth']==1)
        cls.raw=(APP.parent/'capillary_quality_1000/pictures'/cls.file).read_bytes()
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),QuietHandler)
        cls.server.engine=cls.engine;cls.server.inference_token=secrets.token_urlsafe(32)
        cls.server.image_files={cls.file}
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base='http://127.0.0.1:'+str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=5)

    def request(self,path,body=None,headers=None):
        req=urllib.request.Request(self.base+path,data=body,headers=headers or {})
        try:
            with urllib.request.urlopen(req,timeout=90) as r:return r.status,r.read(),r.headers
        except urllib.error.HTTPError as e:return e.code,e.read(),e.headers

    def test_upload_matches_existing_final_model(self):
        output=self.engine.predict(self.raw,['neural','trees_extended','logistic_basic'],mode='precise')
        self.assertTrue(output['known_image']['in_training'])
        gray,bgr,_=decode_upload(self.raw)
        from .features import from_gray
        f=from_gray(gray)
        for row in output['results']:
            b=self.engine.models[row['model']]
            if row['model']=='neural':
                from .neural import embedding_from_bgr
                x=embedding_from_bgr(self.engine.backbone,bgr).reshape(1,-1)
            else:x=np.array([[f[k] for k in b['features']]])
            p,_=predict_bundle(b,x)
            self.assertAlmostEqual(row['p_ok'],float(p[0]),places=10)
            self.assertEqual(row['threshold'],b['thresholds']['precise'])
        self.assertEqual(output['results'][-1]['decision'],'disabled')

    def test_manual_endpoints_and_validation(self):
        for threshold,expected in [(0.,'ok'),(1.,'not_ok')]:
            out=self.engine.predict(self.raw,['logistic_basic'],mode='manual',manual_threshold=threshold)
            self.assertEqual(out['results'][0]['decision'],expected)
        for models in [[],['unknown'],['neural','neural']]:
            with self.assertRaises(ValueError):self.engine.predict(self.raw,models)
        for threshold in [float('nan'),-1,2,None]:
            with self.assertRaises(ValueError):self.engine.predict(self.raw,['neural'],mode='manual',manual_threshold=threshold)

    def test_resize_requires_opt_in(self):
        gray=cv2.imdecode(np.frombuffer(self.raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        small=cv2.resize(gray,(512,384));ok,encoded=cv2.imencode('.png',small);self.assertTrue(ok)
        with self.assertRaises(ValueError):decode_upload(encoded.tobytes())
        a,b,info=decode_upload(encoded.tobytes(),resize=True)
        self.assertEqual(a.shape,(768,1024));self.assertTrue(info['resized'])
        self.assertEqual((info['width'],info['height']),(512,384))
        with self.assertRaises(ValueError):decode_upload(b'not an image')
        _,wrong=cv2.imencode('.png',np.zeros((50,100),np.uint8))
        with self.assertRaises(ValueError):decode_upload(wrong.tobytes(),resize=True)

    def test_http_upload_and_guards(self):
        status,body,_=self.request('/api/inference/config');self.assertEqual(status,200)
        config=json.loads(body);token=config['token'];self.assertEqual(len(config['models']),9)
        status,_,_=self.request('/api/inference/predict?models=logistic_basic',self.raw);self.assertEqual(status,403)
        headers={'X-Inference-Token':token,'Content-Type':'application/octet-stream'}
        status,body,_=self.request('/api/inference/predict?models=logistic_basic&mode=balanced',self.raw,headers)
        self.assertEqual(status,200);self.assertEqual(len(json.loads(body)['results']),1)
        status,_,_=self.request('/api/inference/predict?models=logistic_basic',self.raw,dict(headers,Origin='https://foreign.example'))
        self.assertEqual(status,403)
        status,_,_=self.request('/api/inference/predict?models=missing',self.raw,headers);self.assertEqual(status,400)
        status,_,_=self.request('/api/inference/predict?models=neural',b'invalid',headers);self.assertEqual(status,400)
        for path in ['/upload','/upload.js','/upload.css','/']:
            status,_,h=self.request(path);self.assertEqual(status,200);self.assertIn('nosniff',h['X-Content-Type-Options'])
        status,_,_=self.request('/artifacts/models_features.joblib');self.assertEqual(status,404)


if __name__=='__main__':unittest.main()
