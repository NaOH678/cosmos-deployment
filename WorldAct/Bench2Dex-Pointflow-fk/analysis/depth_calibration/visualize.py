from pathlib import Path
import json,numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;R=json.load(open(B/'results.json'));A=R['aggregate'];selection=R['selection'];labels={2:'Hand',3:'Object',4:'Static region'};models=['raw','offset','scale','affine'];colors=['#777777','#2196f3','#ff9800','#009688']
fig,axs=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
for row,group in enumerate(['sandwich','dagger']):
 ax=axs[row,0]
 for i,model in enumerate(models):
  vals=[next(r for r in A if (r['group'],r['fit_scope'],r['model'],r['split'],r['cls'])==(group,'all_classes',model,'test_episode',cls))['episode_median_abs_depth_mm']['median']/10 for cls in [2,3,4]]
  ax.bar(np.arange(3)+(i-1.5)*.2,vals,width=.2,label=model,color=colors[i])
 ax.set_xticks(np.arange(3),['Hand','Object','Static region']);ax.set(ylabel='Median across episode median |Z error| (cm)',title=f'{group}: 6 entirely held-out episodes');ax.legend(fontsize=9)
 ax=axs[row,1]
 for i,model in enumerate(models):
  vals=[]
  for metric in ['episode_median_native_xyz_mm','episode_median_knownK_xyz_mm']:
   vals.append(next(r for r in A if (r['group'],r['fit_scope'],r['model'],r['split'],r['cls'])==(group,'all_classes',model,'test_episode',2))[metric]['median']/10)
  ax.bar(np.arange(2)+(i-1.5)*.2,vals,width=.2,label=model,color=colors[i])
 ax.set_xticks([0,1],['Original PF rays','Calibrated camera rays']);ax.set(ylabel='Hand 3D discrepancy (cm)',title=f'{group}: depth correction vs depth + intrinsics');ax.legend(fontsize=9)
fig.suptitle('D435 held-out calibration | parameters fitted on separate episodes / early frames only',fontsize=15);fig.savefig(B/'heldout_summary.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
for row,group in enumerate(['sandwich','dagger']):
 for col,cls in enumerate([2,4]):
  ax=axs[row,col]
  for item in [v for v in selection if v['group']==group and v['split']=='test_episode']:
   d=np.load(B/'samples'/group/item['episode']/'points.npz');keep=np.flatnonzero(d['cls']==cls)[::30];ax.scatter(d['pred'][keep,2],d['target'][keep],s=3,alpha=.35,label=item['episode'][8:12])
  lim=[.35,1.35];ax.plot(lim,lim,'k--',label='perfect agreement')
  for model,color in zip(models[1:],colors[1:]):
   p=next(x for x in R['models'] if (x['group'],x['fit_scope'],x['model'])==(group,'all_classes',model));ax.plot(lim,np.array(lim)*p['scale']+p['offset_m'],color=color,label=model)
  ax.set(xlabel='PointFlow depth (m)',ylabel='D435 colour-frame depth (m)',title=group+' / '+labels[cls]);ax.legend(fontsize=7,ncol=2)
fig.savefig(B/'depth_relationship.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
for row,group in enumerate(['sandwich','dagger']):
 item=next(v for v in selection if v['group']==group and v['split']=='test_episode');ep=item['episode'];path=sorted((B/'samples'/group/ep).glob('preview_*.npz'))[-1];frame=int(path.stem.split('_')[1]);d=np.load(path);cap=cv2.VideoCapture(str(Path(item['raw'])/'videos/head.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read();cap.release();assert ok;img=cv2.cvtColor(cv2.resize(img,(640,448)),cv2.COLOR_BGR2RGB);uv=d['uv'];p=next(x for x in R['models'] if (x['group'],x['fit_scope'],x['model'])==(group,'all_classes','offset'));rawerr=(d['pred'][:,2]-d['target'])*100;corrected=(d['pred'][:,2]+p['offset_m']-d['target'])*100
 for col,(value,title) in enumerate([(None,f'{group} ep{ep[8:12]} frame {frame}'),(rawerr,'Raw PF minus D435 (cm)'),(corrected,'Held-out: fixed offset minus D435 (cm)')]):
  ax=axs[row,col];ax.imshow(img)
  if value is None:ax.scatter(uv[:,0],uv[:,1],s=4,c='#00cc88')
  else:plot=ax.scatter(uv[:,0],uv[:,1],s=10,c=value,cmap='coolwarm',vmin=-25,vmax=25);fig.colorbar(plot,ax=ax,shrink=.65)
  ax.set_title(title,fontsize=11);ax.set(xlim=(0,640),ylim=(448,0));ax.axis('off')
fig.suptitle('Examples from held-out episodes; identical matched hand pixels before / after',fontsize=14);fig.savefig(B/'heldout_rgb_examples.png',dpi=150);plt.close(fig)
print('VISUALS_COMPLETE')
