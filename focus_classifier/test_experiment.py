import unittest
import numpy as np
from .features import from_gray, BASIC
from .modeling import group_splits, choose_thresholds, fit_bundle, predict_bundle
from .report import decision_metrics, probability_metrics


class ExperimentTests(unittest.TestCase):
    def test_constant_images_and_fast_features(self):
        for level in (0,127,255):
            a=np.full((64,64),level,np.uint8);full=from_gray(a);fast=from_gray(a,basic_only=True)
            self.assertEqual(fast,{k:full[k] for k in BASIC})
            self.assertTrue(all(np.isfinite(v) for v in full.values()))
            self.assertEqual(full['tenengrad_smooth'],0)
        self.assertEqual(from_gray(np.full((64,64),255,np.uint8))['largest_glare_fraction'],1)

    def test_groups_never_overlap(self):
        groups=np.repeat(np.arange(30).astype(str),4);y=np.tile([0,0,0,1],30)
        for tr,te in group_splits(y,groups,n=5):
            self.assertFalse(set(groups[tr]) & set(groups[te]))
            for a,b in group_splits(y[tr],groups[tr]):
                self.assertFalse(set(groups[tr][a]) & set(groups[tr][b]))

    def test_no_false_high_precision_claim(self):
        t=choose_thresholds(np.array([0]*20+[1]*5),np.full(25,.2))
        self.assertEqual(t['precise'],2)
        d=decision_metrics(np.array([0,1]),np.array([.1,.2]),2.)
        self.assertIsNone(d['precision_ok']);self.assertEqual(d['accepted'],0)

    def test_known_binary_metrics(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.8,.7,.9]);d=decision_metrics(y,p,.5)
        self.assertEqual((d['tn'],d['fp'],d['fn'],d['tp']),(1,1,0,2))
        perfect=probability_metrics(y,y.astype(float))
        self.assertEqual(perfect['auroc'],1);self.assertEqual(perfect['brier'],0)
        self.assertEqual(perfect['ece_ok']['10'],0);self.assertEqual(perfect['ece_top']['10'],0)

    def test_training_bundle_is_finite(self):
        groups=np.repeat(np.arange(30).astype(str),4);y=np.tile([0,0,0,1],30)
        x=(y+.2).reshape(-1,1)
        b=fit_bundle(x,y,groups,'logistic',20261102);p,raw=predict_bundle(b,x)
        self.assertTrue(np.isfinite(p).all());self.assertTrue(np.all(p[y==1]>p[y==0].max()))
        for split in b['inner_patient_splits']:self.assertFalse(set(split['train']) & set(split['valid']))


if __name__=='__main__':unittest.main()
