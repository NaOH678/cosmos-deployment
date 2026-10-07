"""CPU regression tests for physical units at chunk boundaries."""
import ast,importlib.util,unittest,sys
from pathlib import Path
import torch
import torch.nn.functional as F
B=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chunk_geometry',B/'Track4World/track4world/nets/da3_chunk_geometry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
# Load the actual native alignment functions without loading any model weights.
tree=ast.parse((B/'Track4World/track4world/nets/model.py').read_text())
sys.path[:0]=[str(B/'Track4World'),str(B)]
from track4world.utils.geometry_torch import mask_aware_nearest_resize
from track4world.utils.alignment import align_points_scale_xyz_shift
ns={'torch':torch,'F':F,'mask_aware_nearest_resize':mask_aware_nearest_resize,'align_points_scale_xyz_shift':align_points_scale_xyz_shift}
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ['get_aligned_scene_flow_temporal','refine_confidence_with_geometric_consistency']:
  exec(compile(ast.Module(body=[node],type_ignores=[]),'model_alignment','exec'),ns)
class GeometryTests(unittest.TestCase):
 def test_static_target_uses_source_units(self):
  scales=torch.tensor([2.,2.,4.]);metric=torch.full((3,3,2,3),6.)
  normalized=metric/scales[:,None,None,None]
  pairs=torch.stack([normalized[:-1],normalized[1:]],dim=1)
  fixed=g.target_in_source_units(pairs,scales)
  torch.testing.assert_close(fixed[:,0],fixed[:,1])
  torch.testing.assert_close(fixed[0],pairs[0])
  torch.testing.assert_close(pairs[1,1],torch.full_like(pairs[1,1],1.5))
 def test_restore_endpoint_is_source_scaled_not_target_scaled(self):
  s=torch.tensor([2.,4.,3.]);p=torch.ones((1,3,2,3,3));end=torch.full((1,2,2,3,3),1.5)
  poses=torch.eye(4).repeat(3,1,1);poses[:,:3,3]=1
  pp,ee,ww,cc=g.restore_geometry(p,end,p,poses,s)
  torch.testing.assert_close(pp[0,:,0,0,0],s)
  torch.testing.assert_close(ee[0,:,0,0,0],torch.tensor([3.,6.]))
  torch.testing.assert_close(cc[:,:3,3],s[:,None].expand(-1,3))
  torch.testing.assert_close(poses[:,:3,3],torch.ones((3,3)))
 def test_alignment_keeps_static_boundary_static_and_confident(self):
  scales=torch.tensor([2.,2.,4.]);ids=torch.tensor([0,0,1]);h,w=4,5
  metric=torch.tensor([.1,.2,2.]).expand(1,3,h,w,3).clone()
  normalized=metric/scales[None,:,None,None,None]
  endpoints=normalized[:,:-1].clone()
  yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
  flow=torch.stack([xx,yy],dim=-1).float().expand(1,2,h,w,2).clone()
  conf=torch.ones((1,2,h,w,2))
  aligned,c=g.align_pairs_in_source_units(flow,endpoints,normalized,conf,scales,ids,ns['get_aligned_scene_flow_temporal'],ns['refine_confidence_with_geometric_consistency'])
  torch.testing.assert_close(aligned,endpoints)
  torch.testing.assert_close(c,torch.ones((1,2,2,h,w)))
 def test_one_chunk_matches_native_alignment(self):
  torch.manual_seed(17);h,w=4,5
  p=torch.rand(1,3,h,w,3)+1;end=p[:,:-1]+.01
  yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
  flow=torch.stack([xx,yy],dim=-1).float().expand(1,2,h,w,2).clone();conf=torch.ones(1,2,h,w,2)
  fn,cf=ns['get_aligned_scene_flow_temporal'],ns['refine_confidence_with_geometric_consistency']
  expected=fn(flow,end,p,conf,mode='align_dir');expected_c=cf(flow,expected,p,conf)
  actual,actual_c=g.align_pairs_in_source_units(flow,end,p,conf,torch.full((3,),2.),torch.zeros(3,dtype=torch.long),fn,cf)
  torch.testing.assert_close(actual,expected);torch.testing.assert_close(actual_c,expected_c)
 def test_single_frame_tail_does_not_duplicate_pairs(self):
  p=torch.ones(1,3,2,2,3);s=torch.tensor([1.,2.,3.]);ids=torch.arange(3)
  def align(f,e,p,c,mode):return e
  def confidence(f,e,p,c):return c.permute(0,1,4,2,3)
  e,c=g.align_pairs_in_source_units(torch.zeros(1,2,2,2,2),p[:,:-1],p,torch.ones(1,2,2,2,2),s,ids,align,confidence)
  self.assertEqual(e.shape[1],2);self.assertEqual(c.shape[1],2)
if __name__=='__main__':unittest.main()
