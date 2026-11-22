"""CPU-only MoE CLI integration. Synthetic expert inputs; legal-path stand-in replaces GPU trellis search.
Executes the actual main body with only its CUDA placement literal redirected to CPU. Source/dump/reference and
hardware queries are doubles; Hessians, encode/refit, proxies, packed decoding, filesystem/chunks and resume are real.
Does not qualify native encoding, a real checkpoint, router weighting, task quality or timing.
"""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from htr import encode, wire
from htr.scale_refit import refit_packed
from test_quantize_refit_cli import legal_backend

ROOT=Path(__file__).resolve().parents[1]
PIN='1c6dc09'
FIELDS=('qweight','scales','su','sv')
torch.set_num_threads(2)

def source_module(pinned=False):
    text=(subprocess.check_output(['git','-C',str(ROOT),'show',f'{PIN}:htr/moe_encode.py'],text=True)
          if pinned else (ROOT/'htr/moe_encode.py').read_text())
    boundary='dev = "cuda"; os.makedirs(a.out, exist_ok=True)'
    if text.count(boundary)!=1:raise AssertionError('CPU placement boundary moved')
    text=text.replace(boundary,'dev = "cpu"; os.makedirs(a.out, exist_ok=True)')
    m=types.ModuleType('htr._moe_refit_cpu_test');m.__package__='htr'
    exec(compile(text,'<moe_encode placement-only CPU adapter>','exec'),m.__dict__)
    return m

def bits(t):return t.contiguous().view(torch.uint8)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def file_digests(d):return {str(p.relative_to(d)):digest(p) for p in d.rglob('*') if p.is_file()}

class MoeMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():raise AssertionError('test must be device-free')
        rng=torch.Generator().manual_seed(80233)
        cls.gu=(torch.randn(2,1280,128,generator=rng)*.02).bfloat16()
        cls.dn=(torch.randn(2,128,640,generator=rng)*.02).bfloat16()
        cls.x=(torch.randn(32,128,generator=rng)*.4).bfloat16()
        cls.ids=torch.tensor([[0,1]]*32)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.m=source_module();self.oracle=source_module(True)
    def invoke(self,out,extra=(),pinned=False,forbid_device=False,forbid_encode=False,on_device=None):
        m=self.oracle if pinned else self.m
        gu,dn=self.gu,self.dn
        class Source:
            def __init__(self,*a):pass
            def chunk(self,e0,e1):
                if (e0,e1)!=(0,2):raise AssertionError('escaped bounded chunk')
                return gu.clone(),dn.clone()
        class Release:
            bits=[8]*512
            def __init__(self,*a):pass
            def expert(self,e):return gu[e,:640].float(),gu[e,640:].float(),dn[e].float()
        argv=['htr.moe_encode','--dump','SYNTHETIC','--src','SYNTHETIC','--reference','SYNTHETIC',
              '--out',str(out),'--bits','4','--layers','0','--chunk','2','--max-chunks','1','--margin-gb','0']+list(extra)
        recorded=[];real=encode.encode_group
        def counted(jobs,k,**kw):
            if forbid_encode:raise AssertionError('unexpected encoder call')
            result=real(jobs,k,**kw);recorded.append(dict(jobs=jobs,K=k,kwargs=kw,result=result));return result
        with contextlib.ExitStack() as stack:
            for target,name,value in [(m,'SourceExperts',Source),(m.M,'Release',Release),
                (m,'load_dump',lambda *args:(self.x.clone(),self.ids.clone(),1)),
                (encode,'VITERBI',legal_backend),(encode,'encode_group',counted)]:
                stack.enter_context(patch.object(target,name,value))
            stack.enter_context(patch.object(torch.cuda,'get_device_name',side_effect=AssertionError('device reached before rejection') if forbid_device else on_device,return_value='CPU_TEST'))
            stack.enter_context(patch.object(torch.cuda,'empty_cache',return_value=None))
            stack.enter_context(patch.object(sys,'argv',argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            m.main()
        self.assertFalse(torch.cuda.is_initialized())
        return recorded
    def enabled(self,out,extra=()):
        try:return self.invoke(out,('--scale-refit',)+tuple(extra))
        except SystemExit as e:self.fail(f'--scale-refit not implemented: parser exit {e.code}')
    def load(self,out):return torch.load(out/'L0/c000.pt',map_location='cpu',weights_only=False)
    def assert_bits(self,a,b):
        self.assertEqual(a.dtype,b.dtype);self.assertEqual(a.shape,b.shape);self.assertTrue(torch.equal(bits(a),bits(b)))
    def test_real_module_help_exposes_opt_in(self):
        r=subprocess.run([sys.executable,'-B','-m','htr.moe_encode','--help'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--scale-refit',r.stdout)
    def test_default_plan_and_packed_fields_match_pinned(self):
        out=self.root/'same-output'
        before=self.invoke(out,pinned=True);z=self.load(out);plan=(out/'PLAN.json').read_bytes()
        # Separate independent target; compare PLAN apart from out (not part of the existing plan schema).
        new=self.root/'new';after=self.invoke(new);got=self.load(new)
        self.assertEqual(plan,(new/'PLAN.json').read_bytes())
        self.assertEqual([r['kwargs'] for r in before],[r['kwargs'] for r in after])
        self.assertEqual(z['rows'],got['rows']);self.assertEqual(z['format'],got['format'])
        self.assertEqual(set(z),set(got));self.assertNotIn('encoder',got)
        for a,b in zip(z['experts'],got['experts']):
            self.assertEqual(a['what_bf16_sha256'],b['what_bf16_sha256'])
            for role in ('gate','up','down'):
                for field in FIELDS:self.assert_bits(a['packed'][role][field],b['packed'][role][field])
    def test_opt_in_all_widths_matches_direct_refit_and_reopens(self):
        changed=0
        for k in (2,3,4,5):
            with self.subTest(K=k):
                out=self.root/f'k{k}';calls=self.enabled(out,('--bits',str(k),'--damp','0.075'))
                self.assertEqual(len(calls),2)
                plan=json.loads((out/'PLAN.json').read_text());self.assertIs(plan['scale_refit'],True)
                z=self.load(out);self.assertEqual(z['encoder'],dict(scale_refit=True,refit_row_batch=128))
                self.assertTrue(self.m.strip_what(z));self.assertEqual(len(z['experts']),2)
                expected=[]
                for call in calls:
                    self.assertIs(call['kwargs']['scale_refit'],True)
                    self.assertEqual(call['kwargs']['refit_row_batch'],128);self.assertEqual(call['kwargs']['damp'],.075)
                    with patch.object(encode,'VITERBI',legal_backend):
                        base=encode.encode_group(call['jobs'],k,**{p:v for p,v in call['kwargs'].items() if p not in ('scale_refit','refit_row_batch')})
                    fitted=[refit_packed(j['W'],j['H'],r,damp=.075,row_batch=128) for j,r in zip(call['jobs'],base)]
                    for a,b,original in zip(call['result'],fitted,base):
                        for f in FIELDS+('What','Q'):self.assert_bits(a[f],b[f])
                        for f in ('qweight','su','sv'):self.assert_bits(a[f],original[f])
                        changed+=a['scale_refit']['changed_rows']
                    expected.append(fitted)
                for i,e in enumerate(z['experts']):
                    refs=(expected[0][2*i],expected[0][2*i+1],expected[1][i])
                    for role,r in zip(('gate','up','down'),refs):
                        for f in FIELDS:self.assert_bits(e['packed'][role][f],r[f])
                        self.assertEqual(e['what_bf16_sha256'][role],hashlib.sha256(bits(r['What'].bfloat16()).numpy().tobytes()).hexdigest())
                    self.assertEqual(z['rows'][i]['scale_refit'],{role:r['scale_refit'] for role,r in zip(('gate','up','down'),refs)})
        self.assertGreater(changed,0,'vacuous fixture: refit changed no scales')
    def test_resume_same_recipe_does_not_encode_or_rewrite(self):
        out=self.root/'refit';self.enabled(out);before=file_digests(out)
        # All other chunks are marked complete for this synthetic resume-only test.
        # Preserve the actual c000 file; the producer skips the other closed chunk paths.
        for e in range(2,512,2):(out/f'L0/c{e:03d}.pt').write_bytes(b'RESUME-ONLY FIXTURE')
        before=file_digests(out);self.invoke(out,('--scale-refit',),forbid_encode=True)
        self.assertEqual(before,file_digests(out))
    def test_legacy_plan_missing_flag_resumes_default(self):
        out=self.root/'legacy';self.invoke(out,pinned=True)
        for e in range(2,512,2):(out/f'L0/c{e:03d}.pt').write_bytes(b'RESUME-ONLY FIXTURE')
        before=file_digests(out);self.invoke(out,forbid_encode=True);self.assertEqual(before,file_digests(out))
    def test_recipe_mismatch_rejected_before_device(self):
        for stored,flag in ((None,True),(False,True),(True,False),('true',False),(1,True)):
            with self.subTest(stored=stored,requested=flag):
                out=self.root/(str(stored)+'-'+str(flag));out.mkdir()
                plan=dict(bits=4,chunk=2,seed=1234,damp=.01,dump='SYNTHETIC',src='SYNTHETIC',layers='0',format=self.m.FORMAT)
                if stored is not None:plan['scale_refit']=stored
                (out/'PLAN.json').write_text(json.dumps(plan));before=file_digests(out)
                with self.assertRaisesRegex(ValueError,'scale.refit'):
                    self.invoke(out,('--scale-refit',) if flag else (),forbid_device=True)
                self.assertEqual(before,file_digests(out))
    def test_plan_created_by_other_recipe_before_claim_is_rejected(self):
        out=self.root/'raced'
        plan=dict(bits=4,chunk=2,seed=1234,damp=.01,dump='SYNTHETIC',src='SYNTHETIC',layers='0',format=self.m.FORMAT,scale_refit=True)
        def create_competing_plan(*args):
            (out/'PLAN.json').write_text(json.dumps(plan));return 'CPU_TEST'
        with self.assertRaisesRegex(ValueError,'scale.refit'):
            self.invoke(out,forbid_encode=True,on_device=create_competing_plan)
        self.assertEqual(json.loads((out/'PLAN.json').read_text()),plan)
        self.assertFalse((out/'L0/c000.pt').exists())
    def test_atomic_plan_creators_keep_one_complete_record(self):
        from concurrent.futures import ThreadPoolExecutor
        publish=getattr(self.m,'write_plan_if_absent',None)
        if not callable(publish):self.fail('missing no-clobber plan publisher')
        path=self.root/'PLAN.json'
        candidates=[dict(writer=i,scale_refit=bool(i%2),payload=str(i)*20000) for i in range(12)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda value:publish(str(path),value),candidates))
        self.assertIn(json.loads(path.read_text()),candidates)
        self.assertEqual([p.name for p in self.root.iterdir()],['PLAN.json'])
        original=path.read_bytes();publish(str(path),dict(replacement=True))
        self.assertEqual(path.read_bytes(),original)
    def test_orphan_chunks_cannot_be_relabelled_refit(self):
        out=self.root/'orphan';(out/'L0').mkdir(parents=True);(out/'L0/c000.pt').write_bytes(b'legacy')
        before=file_digests(out)
        with self.assertRaisesRegex(ValueError,'scale.refit'):
            self.invoke(out,('--scale-refit',),forbid_device=True)
        self.assertEqual(before,file_digests(out))
    def test_high_bit_opt_in_rejected_without_output_or_device(self):
        for k in (6,8):
            out=self.root/f'bad{k}'
            with self.assertRaises(SystemExit) as e:self.invoke(out,('--scale-refit','--bits',str(k)),forbid_device=True)
            self.assertEqual(e.exception.code,2);self.assertFalse(out.exists())
    def test_high_bit_default_keeps_old_arguments(self):
        for k in (6,8):
            out=self.root/f'base{k}'
            # Stop at the first actual encode boundary; no high-bit trellis CPU workload or result claimed.
            with self.assertRaisesRegex(AssertionError,'unexpected encoder call'):
                self.invoke(out,('--bits',str(k)),forbid_encode=True)
            self.assertNotIn('scale_refit',json.loads((out/'PLAN.json').read_text()))

if __name__=='__main__':unittest.main(verbosity=2)
