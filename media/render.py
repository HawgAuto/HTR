"""Rebuild HTR's typeset hero, explanatory diagram, and evidence graphics.

CPU-only, Pillow required. All chart observations come from ../evidence/.
The source artwork is conceptual; the diagrams/charts are deterministic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONT = HERE / 'fonts/SpaceGrotesk.ttf'
INK = '#101719'
PAPER = '#F2F0E9'
WHITE = '#F4F0E8'
MUTED = '#9BA9AA'
COPPER = '#FF9563'
TEAL = '#70CEC8'
GRID = '#D4D8D3'
W = 1774


def font(size, weight=400):
    f = ImageFont.truetype(str(FONT), size)
    f.set_variation_by_axes([weight])
    return f


def text(draw, xy, value, size=30, color=WHITE, weight=400, anchor='lt'):
    f = font(size, weight)
    box = draw.textbbox(xy, value, font=f, anchor=anchor)
    assert box[0] >= 0 and box[1] >= 0, (value, box)
    assert box[2] <= draw.im.size[0] and box[3] <= draw.im.size[1], (value, box, draw.im.size)
    draw.text(xy, value, font=f, fill=color, anchor=anchor)
    return box


def save(im, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    im.convert('RGB').save(target, format='PNG', optimize=True)


def brand_mark(d, x, y, scale=1.0, color=COPPER):
    # A selected path crossing a small trellis; decorative brand mark.
    nodes = [(x+c*15*scale, y+r*15*scale) for c in range(3) for r in range(3)]
    for c in range(2):
        for r in range(3):
            for r2 in range(3):
                if abs(r-r2) <= 1:
                    d.line((nodes[c*3+r], nodes[(c+1)*3+r2]), fill='#3D5356', width=1)
    d.line([nodes[i] for i in [2,4,6]], fill=color, width=max(2, round(3*scale)))
    for i, (nx, ny) in enumerate(nodes):
        radius=2.5*scale
        d.ellipse((nx-radius,ny-radius,nx+radius,ny+radius), fill=color if i in [2,4,6] else '#719190')


def render_hero(background, target, kind='htr'):
    im=Image.open(background).convert('RGB')
    assert im.size == (1774,887), im.size
    # Gentle left scrim, no rectangular panel obscuring the sculpture.
    overlay=Image.new('RGBA',im.size,(0,0,0,0))
    od=ImageDraw.Draw(overlay)
    for x in range(900):
        a=round(55*max(0,1-x/900)**1.8)
        od.line((x,0,x,887),fill=(6,12,14,a))
    im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
    d=ImageDraw.Draw(im)
    brand_mark(d,94,75,1.25)
    text(d,(152,76),'HAWGAUTO  /  HALO TRELLIS',22,WHITE,600)
    if kind == 'htr':
        text(d,(90,190),'Halo',150,WHITE,650)
        text(d,(83,341),'Trellis',150,COPPER,650)
        text(d,(95,548),'More model.',44,WHITE,450)
        text(d,(95,603),'Fewer weight bytes.',44,WHITE,450)
        text(d,(95,745),'LOW-BIT QUANTIZATION',24,TEAL,600)
        text(d,(95,793),'AMD STRIX HALO  /  GFX1151',22,MUTED,450)
    elif kind == 'mimo':
        text(d,(90,192),'MiMo',140,WHITE,650)
        text(d,(89,341),'9B',150,COPPER,650)
        text(d,(95,538),'Two widths.',43,WHITE,450)
        text(d,(95,592),'One K4 head.',43,WHITE,450)
        text(d,(95,729),'V2.6 DISTILL-QWEN-9B',24,TEAL,600)
        text(d,(95,784),'HTR3 / HTR4  ·  STRIX HALO',23,MUTED,450)
    elif kind == 'qwen':
        text(d,(90,211),'Qwen3.8',102,WHITE,650)
        text(d,(90,337),'Flash Next',83,COPPER,650)
        text(d,(95,527),'Precision,',43,WHITE,450)
        text(d,(95,582),'by tensor.',43,WHITE,450)
        text(d,(95,729),'K5 DENSE  /  K4 HEAD + EXPERTS',21,TEAL,600)
        text(d,(95,784),'HALO TRELLIS  ·  STRIX HALO',23,MUTED,450)
    else:
        raise ValueError(kind)
    save(im,target)


def arrow(d, a, b, color=COPPER, width=4):
    d.line((a,b),fill=color,width=width)
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    wing=[(b[0]-15*math.cos(angle+off),b[1]-15*math.sin(angle+off)) for off in [-.5,.5]]
    d.polygon([b,*wing],fill=color)


def render_method(target):
    im=Image.new('RGB',(W,940),INK);d=ImageDraw.Draw(im)
    text(d,(76,60),'From weights to work.',66,WHITE,600)
    text(d,(80,150),'Three connected parts of the HTR pipeline.',31,MUTED)
    starts=[80,655,1230]
    for x,num,title in zip(starts,['01','02','03'],['Condition','Encode','Execute']):
        d.line((x,234,x+465,234),fill='#3C5052',width=2)
        text(d,(x,264),num,25,COPPER,600)
        text(d,(x+60,255),title,48,WHITE,550)
    # Activation-derived conditioning: matrix -> rotated, distributed structure.
    x0,y0=105,397
    for row in range(6):
        for col in range(6):
            level=(row*11+col*7)%17/16
            color=(int(35+100*level),int(64+110*level),int(65+106*level))
            d.rectangle((x0+col*26,y0+row*26,x0+col*26+20,y0+row*26+20),fill=color)
    arrow(d,(282,478),(330,478))
    cx,cy=418,478
    for row in range(6):
        for col in range(6):
            xx=(col-2.5)*21;yy=(row-2.5)*21
            rx=cx+(xx-yy)*.75;ry=cy+(xx+yy)*.75
            c=TEAL if (row+col)%2 else '#528886'
            d.polygon([(rx,ry-12),(rx+12,ry),(rx,ry+12),(rx-12,ry)],fill=c)
    arrow(d,(565,478),(615,478),color='#536E70',width=3)
    arrow(d,(1140,478),(1190,478),color='#536E70',width=3)
    # Trellis: valid-path illustration, not a specific measured encoding.
    nodes={(c,r):(685+c*95,390+r*65) for c in range(5) for r in range(4)}
    for c in range(4):
        for r in range(4):
            for rr in [r,(r+1)%4]:d.line((nodes[c,r],nodes[c+1,rr]),fill='#344F53',width=2)
    path=[0,1,1,2,3]
    d.line([nodes[c,r] for c,r in enumerate(path)],fill=COPPER,width=5)
    for key,(x,y) in nodes.items():
        selected=path[key[0]]==key[1]
        d.ellipse((x-7,y-7,x+7,y+7),fill=COPPER if selected else '#7DA6A7')
    for j in range(17):
        x=685+j*23;d.rectangle((x,657,x+17,684),fill=COPPER if j%3 else TEAL)
    # Runtime choices. Both reconstruct weights; prefill uses a different path.
    text(d,(1234,380),'SMALL-ROW GENERATION',21,TEAL,600)
    d.rounded_rectangle((1235,426,1395,494),radius=8,outline='#608C8B',width=2)
    text(d,(1315,449),'W1',28,WHITE,600,'mt')
    arrow(d,(1410,460),(1475,460))
    text(d,(1490,442),'GEMV',31,COPPER,600)
    text(d,(1234,546),'LARGER-ROW PREFILL',21,TEAL,600)
    d.rounded_rectangle((1235,590,1395,658),radius=8,outline='#608C8B',width=2)
    text(d,(1315,613),'BF16 W',25,WHITE,500,'mt')
    arrow(d,(1410,624),(1475,624))
    text(d,(1490,606),'GEMM',31,COPPER,600)
    labels=[['Activation-derived Hessian','Randomized Hadamard transforms'],['Block-LDLQ + Viterbi','W1 stream + scales + signs'],['Reconstruct at consumption','Select kernels for the row count']]
    for x,lines in zip(starts,labels):
        text(d,(x,746),lines[0],27,WHITE,450)
        text(d,(x,793),lines[1],25,MUTED)
    text(d,(80,885),'CONCEPTUAL MAP  /  Native serving binds a matching loader, model adapter, and kernel path.',23,MUTED)
    save(im,target)


def codec_rows():
    raw=json.loads((ROOT/'evidence/codec-screen.json').read_text())
    rows=[r for r in raw['rows'] if 'kl' in r]
    assert len({r['label'] for r in rows})==len(rows)
    return raw,rows


def render_codec(target):
    raw,rows=codec_rows()
    im=Image.new('RGB',(W,1120),PAPER);d=ImageDraw.Draw(im)
    text(d,(76,54),'More fidelity per bit.',65,INK,600)
    text(d,(80,144),'Same teacher. Same target tensors. All quantized controls.',30,'#4B5B5D')
    colors={'htr':'#B94D25','gptq':'#216F73','ldlq_scalar_rht':'#7B6191','pt':'#697E7C'}
    legend=[('HTR','htr'),('GPTQ','gptq'),('Rotated scalar LDLQ','ldlq_scalar_rht'),('GGML controls','pt')]
    lx=83
    for name,kind in legend:
        d.ellipse((lx,209,lx+15,224),fill=colors[kind]);text(d,(lx+28,201),name,25,INK,500)
        lx += {'htr':150,'gptq':180,'ldlq_scalar_rht':365,'pt':255}[kind]
    text(d,(1645,203),'Lower KL is closer to the teacher',24,'#4B5B5D',400,'rt')
    left,right,top,bottom=145,1580,305,855
    X=lambda bpw:left+(bpw-1.8)/(4.7-1.8)*(right-left)
    Y=lambda kl:bottom-kl/.5*(bottom-top)
    for k in range(6):
        v=k*.1;y=Y(v);d.line((left,y,right,y),fill=GRID,width=2)
        text(d,(left-20,y),f'{v:.1f}',26,'#536366',400,'rm')
    for value in [2,2.5,3,3.5,4,4.5]:
        x=X(value);d.line((x,top,x,bottom),fill='#DFE1DA',width=1)
        text(d,(x,bottom+26),f'{value:g}',26,'#536366',400,'mt')
    text(d,(left,264),'Teacher KL',25,INK,550)
    text(d,((left+right)/2,935),'Target bits / weight  ·  signs and scales included',29,INK,450,'mt')
    offsets={'htr:2':(22,18),'htr:3':(23,22),'htr:4':(-95,-58),'gptq:3':(-28,-52),'gptq:4':(-28,-64),'ldlq_scalar_rht:3':(-228,-45),'Q4_K':(20,-42),'IQ4_XS':(25,12),'Q3_K':(24,16),'IQ3_XXS':(-163,-33),'Q2_K':(22,-20),'IQ2_XS':(22,-20)}
    derived=[]
    for row in sorted(rows,key=lambda r:r['method']=='htr'):
        x,y=X(row['bpw_quantized']),Y(row['kl']);color=colors[row['method']]
        radius=10 if row['method']=='htr' else 7
        if row['method']=='htr':d.ellipse((x-17,y-17,x+17,y+17),outline='#E2C0AA',width=2)
        d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
        ox,oy=offsets[row['label']]
        text(d,(x+ox,y+oy),row['label'],27,color,600 if row['method']=='htr' else 450)
        derived.append({'label':row['label'],'target_bpw':row['bpw_quantized'],'teacher_kl':row['kl'],'pixel_xy':[x,y]})
    text(d,(80,1015),'MiMo-V2.6-Distill-Qwen-9B  /  64 × 2048 WikiText windows  /  embeddings and head excluded',23,'#536366')
    text(d,(80,1060),'Top-128-plus-rest teacher KL; not task accuracy. Source: evidence/codec-screen.json',23,'#536366')
    save(im,target)
    return derived


def render_serving(target):
    raw=json.loads((ROOT/'evidence/historical-serving.json').read_text())
    rows=raw['rows'];assert len({r['label'] for r in rows})==len(rows)
    im=Image.new('RGB',(W,1150),INK);d=ImageDraw.Draw(im)
    text(d,(76,55),'Generation gains. Prefill tradeoffs.',61,WHITE,600)
    text(d,(80,146),'Radeon 8060S  /  one sequence  /  no speculation',30,MUTED)
    text(d,(80,199),'Bars: median of three repetitions. Whiskers: observed min–max.',26,MUTED)
    gx0,gx1,px0,px1=710,1130,1280,1645
    text(d,(gx0,268),'Wall generation',32,COPPER,550)
    text(d,(gx0,313),'16 in → 256 out',25,MUTED)
    text(d,(px0,268),'Prompt wall',32,TEAL,550)
    text(d,(px0,313),'2048 in → 1 out',25,MUTED)
    XG=lambda v:gx0+v/60*(gx1-gx0)
    XP=lambda v:px0+v/1800*(px1-px0)
    for value in [0,20,40,60]:
        x=XG(value);d.line((x,375,x,915),fill='#2B3B3E',width=1)
        text(d,(x,927),str(value),22,MUTED,400,'mt')
    for value in [0,600,1200,1800]:
        x=XP(value);d.line((x,375,x,915),fill='#2B3B3E',width=1)
        text(d,(x,927),str(value),22,MUTED,400,'mt')
    derived=[]
    for idx,row in enumerate(rows):
        y=405+idx*93
        selected=row['label'] in {'mimo9b-htr4-lmh4-final','mimo9b-htr3-lmh4-final'}
        if selected:d.rounded_rectangle((65,y-34,1706,y+40),radius=5,fill='#1C292B')
        text(d,(82,y),row['label'],27,WHITE if selected else '#AFBDBD',550 if selected else 400,'lm')
        medg=statistics.median(row['tg_tok_s']);medp=statistics.median(row['pp_tok_s'])
        for values,med,X,start,color,decimals in [(row['tg_tok_s'],medg,XG,gx0,COPPER,3),(row['pp_tok_s'],medp,XP,px0,TEAL,2)]:
            fill=color if selected else ('#8F715F' if start==gx0 else '#547B7B')
            if idx==0:fill='#727D7D'
            d.rectangle((start,y-15,X(med),y+15),fill=fill)
            lo,hi=X(min(values)),X(max(values))
            d.line((lo,y,hi,y),fill=WHITE,width=3)
            for xx in [lo,hi]:d.line((xx,y-9,xx,y+9),fill=WHITE,width=2)
            text(d,(max(X(med),hi)+10,y),f'{med:.{decimals}f}',25,WHITE,500,'lm')
        derived.append({'label':row['label'],'wall_generation_tok_s':medg,'prompt_wall_tok_s':medp,'generation_repetitions':row['tg_tok_s'],'prompt_repetitions':row['pp_tok_s']})
    text(d,(gx0,976),'TOKENS / SECOND',21,MUTED,500)
    text(d,(px0,976),'TOKENS / SECOND',21,MUTED,500)
    text(d,(80,1032),'Historical E10 / MiMo-9B. Timings include the complete in-process generate() call, including prompt time.',23,MUTED)
    text(d,(80,1080),'K3-head row is diagnostic. Source: evidence/historical-serving.json; original labels and all repetitions retained.',23,MUTED)
    save(im,target)
    return derived


def render_all(out):
    out.mkdir(parents=True,exist_ok=True)
    render_hero(HERE/'source/trellis-halo.webp',out/'hero.png')
    render_method(out/'method-map.png')
    codec=render_codec(out/'codec-fidelity.png')
    serving=render_serving(out/'serving-tradeoff.png')
    data={'scope':'Offline visualizations of retained experiments, not new measurements.',
          'source_sha256':{name:hashlib.sha256((ROOT/'evidence'/name).read_bytes()).hexdigest() for name in ['codec-screen.json','historical-serving.json']},
          'codec':codec,'serving':serving}
    (out/'chart-data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE)
    args=p.parse_args()
    data=render_all(args.out)
    print(json.dumps({'output':str(args.out),'codec_rows':len(data['codec']),'serving_rows':len(data['serving']),'images':['hero.png','method-map.png','codec-fidelity.png','serving-tradeoff.png']}))
