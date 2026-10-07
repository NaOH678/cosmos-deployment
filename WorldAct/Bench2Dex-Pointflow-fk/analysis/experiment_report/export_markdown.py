from pathlib import Path
from html.parser import HTMLParser
import base64,re,zipfile
root=Path('/mnt/afs/Bench2Dex/analysis/experiment_report')
out=root/'PointFlow_FK_实验总结';out.mkdir(exist_ok=True)
assets=out/'附件';assets.mkdir(exist_ok=True)
class Convert(HTMLParser):
 def __init__(self):
  super().__init__();self.parts=[];self.active=False;self.skip=0;self.link=None;self.row=None;self.cell=None;self.rows=[];self.n=0;self.plot=0
 def put(self,s):
  if self.cell is not None:self.cell.append(s)
  else:self.parts.append(s)
 def save(self,uri,name):
  (assets/name).write_bytes(base64.b64decode(uri.split(',',1)[1]));return '附件/'+name
 def handle_starttag(self,t,attrs):
  d=dict(attrs)
  if t=='main':self.active=True;return
  if not self.active:return
  if t in ['nav','script','style']:self.skip+=1;return
  if self.skip:return
  if t=='table':self.rows=[]
  elif t=='tr':self.row=[]
  elif t in ['td','th']:self.cell=[]
  elif t in ['h1','h2','h3']:self.put('\n\n'+'#'*int(t[1])+' ')
  elif t in ['p','div','figure','details','ul','pre','figcaption','summary']:self.put('\n\n')
  elif t=='li':self.put('\n- ')
  elif t in ['b','strong']:self.put('**')
  elif t=='code':self.put('`')
  elif t=='br':self.put('\n\n')
  elif t in ['img','video']:
   self.n+=1;uri=d['src'];mime=uri.split(';')[0];ext='mp4' if t=='video' else ('png' if 'png' in mime else 'jpg');name=f'{self.n:02d}.{ext}';path=self.save(uri,name)
   self.put(f'\n\n'+('!' if t=='img' else '')+f'[{d.get("alt","播放视频（MP4）")}]({path})\n\n')
  elif t=='a':
   href=d.get('href','')
   if href.startswith('data:application/json'):
    path=self.save(href,d['download']);self.link=path;self.put('[')
   else:self.link=None
  elif t=='button':
   self.plot+=1;name=f'交互三维_{self.plot}.html';self.save('data:text/html;base64,'+d['data-content'],name);self.put(f'[可选交互三维附件 {self.plot}](附件/{name})');self.skip+=1
 def handle_endtag(self,t):
  if t=='main':self.active=False;return
  if not self.active:return
  if t in ['nav','script','style','button'] and self.skip:self.skip-=1;return
  if self.skip:return
  if t in ['td','th']:
   self.row.append(''.join(self.cell).strip().replace('|','\\|').replace('\n',' '));self.cell=None
  elif t=='tr':self.rows.append(self.row);self.row=None
  elif t=='table':
   self.put('\n\n'+'\n'.join('| '+' | '.join(r)+' |'+ ('\n| '+' | '.join(['---']*len(r))+' |' if i==0 else '') for i,r in enumerate(self.rows))+'\n\n')
  elif t in ['b','strong']:self.put('**')
  elif t=='code':self.put('`')
  elif t=='a' and self.link:self.put(']('+self.link+')');self.link=None
  elif t in ['h1','h2','h3','p','figure','figcaption','summary','pre']:self.put('\n\n')
 def handle_data(self,s):
  if self.active and not self.skip:self.put(s)
c=Convert();c.feed((root/'pointflow_fk_experiment_report.html').read_text());md=''.join(c.parts)
md=md.replace('单文件离线报告 · 图片、5 段回放、3 个交互三维及关键结果 JSON 均内嵌','Markdown 报告 · 附件含 12 张图片、5 段回放、3 个可选交互三维及关键结果 JSON')
md=md.replace('以下三维视图按需在本文件内加载，可旋转缩放；不依赖原目录。它们展示已有抽样结果，不是新的实验。','以下交互三维作为可选附件保留，用浏览器打开可旋转缩放；正文为 Markdown，阅读正文不需要打开 HTML。它们展示已有抽样结果，不是新的实验。')
md=md.replace('路径用于追溯本次环境，不是报告展示所需的外部依赖。下载附件保留关键数值、样本选择及检查记录；体积较大的逐点原始结果仍保留在实验目录。','路径用于追溯实验环境。附件目录保留图片、视频、关键数值、样本选择及检查记录；体积较大的逐点结果仍保留在实验目录。移动报告时请同时保留附件目录。')
md=md.replace('所有报告媒体离线内嵌；浏览器需支持 MP4/H.264。图片可点击放大，视频可全屏播放。','所有报告媒体位于附件目录；视频为 MP4/H.264 格式，可用本地播放器打开。')
md=re.sub(r'\n{3,}','\n\n',md).strip()+'\n'
p=out/'实验总结.md';p.write_text(md)
links=re.findall(r'\]\((附件/[^)]+)\)',md)
assert all((out/x).is_file() for x in links)
z=root/'PointFlow_FK_实验总结.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for path in sorted(out.rglob('*')):
  if path.is_file():f.write(path,path.relative_to(root))
print(p);print(z);print('Validated attachment links:',len(links),'ZIP bytes:',z.stat().st_size)
