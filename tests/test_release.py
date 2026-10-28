"""CPU release tests. No GPU initialization, model loads, or external datasets."""
import importlib.util
import pathlib
import unittest
import torch
from htr import encode, wire

torch.set_num_threads(2)
ROOT=pathlib.Path(__file__).resolve().parents[1]

class PackagingTests(unittest.TestCase):
    def test_reference_encoder_is_self_contained(self):
        self.assertIsNotNone(importlib.util.find_spec('htr.reference'), 'reference Viterbi must ship inside htr')
        x=torch.linspace(-1,1,256)[None]
        s=encode._viterbi_ref(x,torch.ones_like(x),wire.codebook_f32(),4)
        self.assertEqual(tuple(s.shape),(1,256))
        self.assertTrue(torch.equal(s[:,1:]>>4,s[:,:-1]&4095))
        self.assertTrue(torch.equal(s[:,0]>>4,s[:,-1]&4095))
        self.assertTrue(torch.equal(s,wire.unpack_states(wire.pack_states(s,4),4)))

    def test_no_operator_specific_paths_in_package(self):
        bad=[]
        for p in (ROOT/'htr').rglob('*'):
            if p.suffix in ('.py','.cpp','.cu'):
                if '/home/' in p.read_text() or '/mnt/' in p.read_text():bad.append(str(p.relative_to(ROOT)))
        self.assertEqual(bad,[], 'distribution must not require operator paths')

    def test_each_supported_width_decodes_with_independent_bit_oracle(self):
        for k in (2,3,4,5):
            g=torch.Generator().manual_seed(k)
            words=torch.randint(-(2**31),2**31,(2,8*k),dtype=torch.int32,generator=g)
            states=wire.unpack_states(words,k)
            expected=[]
            for row in words.tolist():
                bits=''.join(format(w&0xffffffff,'032b') for w in row)
                expected.append([int(''.join(bits[(k*(t+1)-16+j)%len(bits)] for j in range(16)),2) for t in range(256)])
            self.assertTrue(torch.equal(states,torch.tensor(expected)))
            self.assertTrue(torch.equal(words,wire.pack_states(states,k)))

    def test_sign_convention(self):
        signs=torch.tensor([1.,-1.,1.,-1.,1.,-1.,1.,-1.])
        self.assertEqual(wire.pack_signs(signs).item(),170)
        self.assertTrue(torch.equal(signs,wire.unpack_signs(wire.pack_signs(signs),8)))

    def test_hadamard_inverse(self):
        h=wire.hadamard(128,'cpu')
        self.assertTrue(torch.allclose(h@h.T,torch.eye(128),atol=2e-6))

    def test_native_sources_are_in_package(self):
        for f in ['htr_kernels.cu','htr_kernels.cpp']:
            self.assertGreater((ROOT/'htr/csrc'/f).stat().st_size,100)

class EncoderRoundtripTests(unittest.TestCase):
    def test_cpu_matrix_encode_save_reopen_and_decode(self):
        from safetensors.torch import save_file,load_file
        import tempfile
        self.assertIsNotNone(importlib.util.find_spec('htr.reference'),'missing distributable reference')
        g=torch.Generator().manual_seed(12)
        w=torch.randn(128,128,generator=g)*0.02
        h=torch.eye(128)
        old=encode.VITERBI
        try:
            encode.VITERBI=None
            result=encode.encode_group([{'W':w,'H':h}],4,max_batch=8)[0]
        finally:encode.VITERBI=old
        packed={k:result[k] for k in ['qweight','scales','su','sv']}
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp)/'matrix.safetensors';save_file(packed,str(p));r=load_file(str(p))
            actual=wire.dequant(r['qweight'],r['scales'],r['su'],r['sv'])
        self.assertTrue(torch.equal(actual,result['What']))
        self.assertEqual(tuple(result['qweight'].shape),(8,8,32))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(((actual-w).square().sum()/w.square().sum()).item(),0.02)

if __name__=='__main__':unittest.main()
