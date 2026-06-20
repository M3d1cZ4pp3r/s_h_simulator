import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

const SR = 44100;
const N  = 32768;
const MAX_F = 20000;
const LOG_LO = Math.log10(300);
const LOG_HI = Math.log10(SR);

// ════════════════════════════════════════
//  FFT CORE  (shared by analysis + idealLP)
// ════════════════════════════════════════
function fftInPlace(re, im) {
  const n = re.length;
  for (let i=1,j=0; i<n; i++) {
    let b=n>>1; for(;j&b;b>>=1) j^=b; j^=b;
    if(i<j){ let t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t; }
  }
  for(let len=2;len<=n;len<<=1){
    const ang=-2*Math.PI/len, wr=Math.cos(ang), wi=Math.sin(ang);
    for(let i=0;i<n;i+=len){
      let cr=1,ci=0;
      for(let j=0;j<(len>>1);j++){
        const h=len>>1, ur=re[i+j],ui=im[i+j];
        const vr=re[i+j+h]*cr-im[i+j+h]*ci, vi=re[i+j+h]*ci+im[i+j+h]*cr;
        re[i+j]=ur+vr; im[i+j]=ui+vi; re[i+j+h]=ur-vr; im[i+j+h]=ui-vi;
        const nr=cr*wr-ci*wi; ci=cr*wi+ci*wr; cr=nr;
      }
    }
  }
}

function computeSpectrum(sig) {
  const n=sig.length, re=new Float64Array(n), im=new Float64Array(n);
  for(let i=0;i<n;i++) re[i]=sig[i]*0.5*(1-Math.cos(2*Math.PI*i/(n-1)));
  fftInPlace(re,im);
  const mag=new Float32Array(n>>1), sc=2/(n*0.5);
  for(let i=0;i<(n>>1);i++) mag[i]=Math.sqrt(re[i]*re[i]+im[i]*im[i])*sc;
  return mag;
}

// ════════════════════════════════════════
//  OSCILLATOR
// ════════════════════════════════════════
function osc(type,f,A,len,sr){
  const x=new Float32Array(len);
  for(let i=0;i<len;i++){
    const p=(f*i/sr)%1;
    x[i]=A*(type==='sin'?Math.sin(2*Math.PI*p):type==='sq'?(p<.5?1:-1):type==='saw'?(2*p-1):(1-4*Math.abs(p-.5)));
  }
  return x;
}

// ════════════════════════════════════════
//  SAMPLE-AND-HOLD
// ════════════════════════════════════════
function applySaH(inp,sr,fs,droopTau,aperFrac){
  const out=new Float32Array(inp.length), period=sr/fs;
  const aperSamp=Math.max(1,Math.round(aperFrac*period));
  let held=0,next=0,age=1e9;
  for(let i=0;i<inp.length;i++){
    if(i>=next){age=0;next+=period;}
    if(age<aperSamp) held=inp[i];
    else if(droopTau>0) held*=Math.exp(-1/(droopTau*sr));
    age++; out[i]=held;
  }
  return out;
}

// ════════════════════════════════════════
//  LOW-PASS FILTERS
// ════════════════════════════════════════
function rcLP(inp,sr,fc){
  const a=2*Math.PI*fc/(2*Math.PI*fc+sr), out=new Float32Array(inp.length);
  out[0]=inp[0];
  for(let i=1;i<inp.length;i++) out[i]=a*inp[i]+(1-a)*out[i-1];
  return out;
}

function bqLP(inp,sr,fc,Q){
  const w=2*Math.PI*fc/sr, cw=Math.cos(w), sw=Math.sin(w);
  const a=sw/(2*Q), d=1/(1+a);
  const b0=(1-cw)/2*d, b1=(1-cw)*d, b2=(1-cw)/2*d, a1=-2*cw*d, a2=(1-a)*d;
  const out=new Float32Array(inp.length);
  let x1=0,x2=0,y1=0,y2=0;
  for(let i=0;i<inp.length;i++){
    const y=b0*inp[i]+b1*x1+b2*x2-a1*y1-a2*y2;
    [x2,x1,y2,y1]=[x1,inp[i],y1,y]; out[i]=y;
  }
  return out;
}

// Butterworth 4th order — two Biquad stages
// Q-Werte aus: Q_k = 1/(2*cos((2k-1)π/8)), k=1,2
function bw4LP(inp,sr,fc){
  return bqLP(bqLP(inp,sr,fc,0.5412),sr,fc,1.3066);
}

// Butterworth 8th order — four Biquad stages
// Q-Werte: Q_k = 1/(2*cos((2k-1)π/16)), k=1..4
function bw8LP(inp,sr,fc){
  let s=bqLP(inp,  sr,fc,0.5098);
  s    =bqLP(s,    sr,fc,0.6013);
  s    =bqLP(s,    sr,fc,0.8999);
  return bqLP(s,   sr,fc,2.5628);
}

// Ideal / Brickwall — FFT → Frequenz-Maske → IFFT
// Hinweis: Gibbs-Ringing an Blockkanten ist physikalisch korrekt für einen Idealfilter
function idealLP(inp,sr,fc){
  const n=inp.length;
  const re=Float64Array.from(inp), im=new Float64Array(n);
  fftInPlace(re,im);
  const cut=Math.round(fc/sr*n);
  for(let i=cut+1;i<n-cut;i++){re[i]=0;im[i]=0;}
  for(let i=0;i<n;i++) im[i]=-im[i]; // IFFT = conj → FFT → conj → /n
  fftInPlace(re,im);
  const out=new Float32Array(n);
  for(let i=0;i<n;i++) out[i]=re[i]/n;
  return out;
}

function applyLPF(inp,sr,type,fc){
  const f=Math.min(fc,sr*0.499);
  if(type==='rc1')   return rcLP(inp,sr,f);
  if(type==='bw2')   return bqLP(inp,sr,f,0.7071);
  if(type==='sk2')   return bqLP(inp,sr,f,0.9565);
  if(type==='bw4')   return bw4LP(inp,sr,f);
  if(type==='bw8')   return bw8LP(inp,sr,f);
  if(type==='ideal') return idealLP(inp,sr,f);
  return inp.slice();
}

// ════════════════════════════════════════
//  TRANSFER FUNCTIONS (Bode)
// ════════════════════════════════════════
const zohTF=(f,fs)=>{if(!f)return 1;const x=Math.PI*f/fs;return Math.abs(Math.sin(x)/x);};

const lpfTF=(f,type,fc)=>{
  const u=f/fc;
  if(type==='rc1')   return 1/Math.sqrt(1+u*u);
  if(type==='bw2')   return 1/Math.sqrt(1+u**4);
  if(type==='sk2')   {const e=Math.sqrt(10**.1-1),T2=2*u*u-1; return 1/Math.sqrt(1+e*e*T2*T2);}
  if(type==='bw4')   return 1/Math.sqrt(1+u**8);   // Butterworth N=4: (f/fc)^(2N)
  if(type==='bw8')   return 1/Math.sqrt(1+u**16);  // Butterworth N=8
  if(type==='ideal') return u<=1?1:1e-6;            // Heaviside-Sprung
  return 1;
};

const dB=v=>20*Math.log10(Math.max(Math.abs(v),1e-9));

// ════════════════════════════════════════
//  SPEKTRALE PEAK-ANALYSE
// ════════════════════════════════════════
const TYPE_ORDER={fund:0,harm:1,alias:2,zoh:3};

function computePeaks(wtype,f0,fs,maxF){
  const ny=fs/2, seen=new Map();
  const add=(f,label,type,color)=>{
    const k=Math.round(f);
    if(f>1&&f<=maxF){const ex=seen.get(k);if(!ex||(TYPE_ORDER[type]??4)<(TYPE_ORDER[ex.type]??4))seen.set(k,{f,label,type,color});}
  };
  const SLIM=Math.min(fs*6,SR/2-1);
  const step=wtype==='sin'?SLIM:(wtype==='sq'||wtype==='tri')?2:1;
  for(let k=1;k*f0<=SLIM;k+=step){
    const fh=k*f0, kL=k===1?'f₀':`${k}f₀`;
    if(fh<=ny){
      add(fh,k===1?'Grundton':`${k}. Oberton`,k===1?'fund':'harm',k===1?'#60a5fa':'#34d399');
      for(let n=1;n*fs<=maxF+fh;n++){
        const fu=n*fs+fh, fl=n*fs-fh;
        if(fu>ny&&fu<=maxF) add(fu,`ZOH ${n}·fs+${kL}`,'zoh','#c084fc');
        if(fl>ny&&fl<=maxF) add(fl,`ZOH ${n}·fs−${kL}`,'zoh','#c084fc');
      }
    } else {
      let fa=fh%fs; if(fa>ny) fa=fs-fa;
      if(fa>1) add(fa,`Alias (${kL}=${Math.round(fh)}Hz)`,'alias','#f87171');
      for(let n=1;n*fs<=maxF+fh;n++){
        const fu=n*fs+fh, fl=n*fs-fh;
        if(fu>ny&&fu<=maxF) add(fu,`ZOH ${n}·fs+${kL}`,'zoh','#c084fc');
        if(fl>ny&&fl<=maxF) add(fl,`ZOH ${n}·fs−${kL}`,'zoh','#c084fc');
      }
    }
    if(wtype==='sin') break;
  }
  return Array.from(seen.values()).sort((a,b)=>a.f-b.f);
}

// ════════════════════════════════════════
//  AUDIO
// ════════════════════════════════════════
function createPlayer(sig,sr){
  const ctx=new(window.AudioContext||window.webkitAudioContext)({sampleRate:sr});
  const buf=ctx.createBuffer(1,sig.length,sr), ch=buf.getChannelData(0);
  for(let i=0;i<sig.length;i++) ch[i]=Math.max(-1,Math.min(1,sig[i]));
  const src=ctx.createBufferSource(); src.buffer=buf; src.loop=true;
  src.connect(ctx.destination); src.start();
  return{ctx,src};
}

// ════════════════════════════════════════
//  UI-KOMPONENTEN
// ════════════════════════════════════════
const MO={fontFamily:'var(--font-mono)'};
const GR='var(--color-border-tertiary)';
const AX='var(--color-text-secondary)';

function Slider({label,value,unit='',min,max,step=1,onChange,accent='#60a5fa'}){
  return(
    <div style={{marginBottom:10}}>
      <div style={{display:'flex',justifyContent:'space-between',marginBottom:3}}>
        <span style={{fontSize:11,color:AX,...MO}}>{label}</span>
        <span style={{fontSize:12,fontWeight:500,color:'var(--color-text-primary)',...MO}}>{step<1?value.toFixed(2):value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e=>onChange(step<1?parseFloat(e.target.value):parseInt(e.target.value))}
        style={{width:'100%',accentColor:accent}}/>
    </div>
  );
}

function Panel({title,accent='var(--color-text-secondary)',children}){
  return(
    <div style={{background:'var(--color-background-primary)',border:'0.5px solid var(--color-border-tertiary)',borderRadius:8,marginBottom:8,overflow:'hidden'}}>
      <div style={{padding:'6px 12px',fontSize:10,fontWeight:500,color:accent,background:'var(--color-background-secondary)',borderBottom:'0.5px solid var(--color-border-tertiary)',...MO,textTransform:'uppercase',letterSpacing:'0.07em'}}>{title}</div>
      <div style={{padding:'12px 12px'}}>{children}</div>
    </div>
  );
}

function ChkBox({label,checked,onChange,color='#60a5fa',note}){
  return(
    <div style={{marginBottom:8}}>
      <label style={{display:'flex',alignItems:'center',gap:6,fontSize:12,cursor:'pointer',color:'var(--color-text-primary)',...MO}}>
        <input type="checkbox" checked={checked} onChange={e=>onChange(e.target.checked)} style={{accentColor:color}}/>{label}
      </label>
      {note&&<div style={{fontSize:10,color:AX,marginTop:2,marginLeft:18,lineHeight:1.5}}>{note}</div>}
    </div>
  );
}

function TipBox({active,payload,label,fmtLabel=l=>l,fmtVal=v=>v.toFixed(3)}){
  if(!active||!payload?.length) return null;
  return(
    <div style={{background:'var(--color-background-secondary)',border:'0.5px solid var(--color-border-secondary)',borderRadius:6,padding:'6px 10px',fontSize:11,...MO}}>
      <div style={{color:AX,marginBottom:3}}>{fmtLabel(label)}</div>
      {payload.map((p,i)=><div key={i} style={{color:p.stroke||p.color}}>{p.name||p.dataKey}: {fmtVal(p.value)}</div>)}
    </div>
  );
}

function TabBtn({active,onClick,children}){
  return(<button onClick={onClick} style={{padding:'9px 14px',fontSize:12,cursor:'pointer',border:'none',background:active?'var(--color-background-primary)':'transparent',color:active?'var(--color-text-primary)':AX,borderBottom:active?'2px solid #60a5fa':'2px solid transparent',...MO}}>{children}</button>);
}

// Filter-Namen für Dropdown
const FILT={
  rc1:'RC 1. Ordnung (−6 dB/Okt)',
  bw2:'Butterworth 2. Ordnung (−12 dB/Okt)',
  sk2:'Sallen-Key / Tschebyscheff 1 dB (resonant)',
  bw4:'Butterworth 4. Ordnung (−24 dB/Okt) ★',
  bw8:'Butterworth 8. Ordnung (−48 dB/Okt) ★★',
  ideal:'Idealfilter / Brickwall (Referenz)',
};

// ════════════════════════════════════════
//  HAUPT-APP
// ════════════════════════════════════════
export default function SaHSimulator(){
  const [wtype,setWtype]=useState('sin');
  const [freq, setFreq] =useState(440);
  const [amp,  setAmp]  =useState(0.8);
  const [fs,   setFs]   =useState(8000);
  const [fsText,setFsText]=useState('8000');
  const [lpfOn,setLpfOn]=useState(true);
  const [lpfType,setLpfType]=useState('bw2');
  const [lpfCut,setLpfCut]=useState(3800);
  const [efxOpen,setEfxOpen]=useState(false);
  const [droopOn,setDroopOn]=useState(false);
  const [droopTau,setDroopTau]=useState(50);
  const [aperOn,setAperOn]=useState(false);
  const [aperPct,setAperPct]=useState(5);
  const [tab,setTab]=useState('time');
  const [inPlay,setInPlay]=useState(false);
  const [outPlay,setOutPlay]=useState(false);
  const [showSaH,setShowSaH]=useState(true);
  const inRef=useRef(null), outRef=useRef(null);

  useEffect(()=>()=>{
    try{inRef.current?.src.stop();inRef.current?.ctx.close();}catch(_){}
    try{outRef.current?.src.stop();outRef.current?.ctx.close();}catch(_){}
  },[]);

  const ny=fs/2, aliasingOn=freq>ny, fsLog=Math.log10(fs);

  // ── Signale ──
  const{inSig,sahSig,outSig}=useMemo(()=>{
    const inSig=osc(wtype,freq,amp,N,SR);
    const tau=droopOn?droopTau/1000:0, aper=aperOn?aperPct/100:0.001;
    const sahSig=applySaH(inSig,SR,fs,tau,aper);
    const outSig=lpfOn?applyLPF(sahSig,SR,lpfType,lpfCut):sahSig.slice();
    return{inSig,sahSig,outSig};
  },[wtype,freq,amp,fs,droopOn,droopTau,aperOn,aperPct,lpfOn,lpfType,lpfCut]);

  // ── Zeitbereich ──
  const timeData=useMemo(()=>{
    const nS=Math.min(Math.ceil(SR/freq*4),10000), step=Math.max(1,Math.floor(nS/1500));
    const data=[];
    for(let i=0;i<nS;i+=step){
      const o={t:+(i/SR*1000).toFixed(4),Ein:+inSig[i].toFixed(4)};
      if(showSaH) o['S&H']=+sahSig[i].toFixed(4);
      o.Aus=+outSig[i].toFixed(4); data.push(o);
    }
    return data;
  },[inSig,sahSig,outSig,freq,showSaH]);

  // ── Spektren ──
  const{specComb,pkList}=useMemo(()=>{
    const fr=SR/N, mb=Math.floor(MAX_F/fr), step=Math.max(1,Math.floor(mb/1200));
    const iM=computeSpectrum(inSig), sM=computeSpectrum(sahSig), oM=computeSpectrum(outSig);
    const rows=[];
    for(let i=1;i<mb;i+=step){
      const f=+(i*fr).toFixed(1);
      const r={f,'Eingang':+dB(iM[i]).toFixed(1)};
      if(showSaH) r['S&H']=+dB(sM[i]).toFixed(1);
      r['Ausgang']=+dB(oM[i]).toFixed(1); rows.push(r);
    }
    return{specComb:rows,pkList:computePeaks(wtype,freq,fs,MAX_F)};
  },[inSig,sahSig,outSig,wtype,freq,fs,showSaH]);

  // ── Übertragungsfunktion ──
  const tfData=useMemo(()=>{
    const mF=Math.min(MAX_F,fs*2.5);
    return Array.from({length:700},(_,i)=>{
      const f=Math.max(1,(i+1)/700*mF);
      const z=zohTF(f,fs), l=lpfOn?lpfTF(f,lpfType,lpfCut):1;
      return{f:+f.toFixed(1),ZOH:+dB(z).toFixed(2),LPF:+dB(l).toFixed(2),Gesamt:+dB(z*l).toFixed(2)};
    });
  },[fs,lpfOn,lpfType,lpfCut]);

  const selPeaks=useMemo(()=>[
    ...pkList.filter(p=>p.type==='fund'||p.type==='harm').slice(0,8),
    ...pkList.filter(p=>p.type==='alias').slice(0,5),
    ...pkList.filter(p=>p.type==='zoh').slice(0,5),
  ],[pkList]);

  const togPlay=useCallback((sig,ref,pl,setP)=>{
    try{ref.current?.src.stop();ref.current?.ctx.close();}catch(_){}
    ref.current=null;
    if(!pl){ref.current=createPlayer(sig,SR);setP(true);}else setP(false);
  },[]);

  const setFsFromLog=v=>{const hz=Math.round(10**parseFloat(v));setFs(hz);setFsText(String(hz));};
  const handleFsText=e=>{setFsText(e.target.value);const v=parseFloat(e.target.value);if(!isNaN(v)&&v>=300&&v<=SR)setFs(Math.round(v));};

  const SEL={background:'var(--color-background-secondary)',border:'0.5px solid var(--color-border-secondary)',borderRadius:6,color:'var(--color-text-primary)',padding:'5px 8px',fontSize:12,width:'100%',cursor:'pointer',...MO};

  // Tick-Styles für Recharts
  const TK13={fontSize:13,...MO};
  const TK12={fontSize:12,...MO};

  return(
    <div style={{display:'flex',flexDirection:'column',height:'100vh'}}>

      {/* Header */}
      <div style={{background:'var(--color-background-secondary)',borderBottom:'0.5px solid var(--color-border-tertiary)',padding:'8px 16px',display:'flex',alignItems:'center',gap:12}}>
        <span style={{fontWeight:500,fontSize:15,...MO}}>S&amp;H Simulator</span>
        <span style={{color:AX,fontSize:11}}>555 → Analogschalter → C + OPV-Buffer → Tiefpass</span>
        {aliasingOn&&<span style={{marginLeft:'auto',background:'var(--color-background-danger)',border:'0.5px solid var(--color-border-danger)',color:'var(--color-text-danger)',fontSize:11,padding:'2px 8px',borderRadius:4,...MO}}>⚠ Aliasing: f₀={freq}Hz &gt; Nyquist={ny}Hz</span>}
      </div>

      <div style={{display:'flex',flex:1,overflow:'hidden'}}>

        {/* ── Controls ── */}
        <div style={{width:274,flexShrink:0,background:'var(--color-background-secondary)',borderRight:'0.5px solid var(--color-border-tertiary)',overflowY:'auto',padding:10}}>

          <Panel title="🎵 Eingang (Oszillator)" accent="#60a5fa">
            <div style={{display:'flex',gap:3,marginBottom:10}}>
              {[['sin','Sinus'],['sq','Rechteck'],['saw','Sägezahn'],['tri','Dreieck']].map(([v,l])=>(
                <button key={v} onClick={()=>setWtype(v)} style={{flex:1,fontSize:9,padding:'4px 0',borderRadius:4,cursor:'pointer',...MO,border:`0.5px solid ${wtype===v?'#3b82f6':'var(--color-border-tertiary)'}`,background:wtype===v?'#1d4ed8':'var(--color-background-tertiary)',color:wtype===v?'#fff':AX}}>
                  {l}
                </button>
              ))}
            </div>
            <Slider label="Frequenz" value={freq} unit=" Hz" min={50}  max={8000} step={1}    onChange={setFreq}/>
            <Slider label="Amplitude" value={amp}  min={0.1} max={1.0}  step={0.01} onChange={setAmp}/>
          </Panel>

          <Panel title="📡 Abtastung (S&H)" accent="#f97316">
            <div style={{fontSize:11,color:AX,marginBottom:4,...MO}}>Abtastrate (logarithmisch)</div>
            <div style={{display:'flex',gap:6,alignItems:'center',marginBottom:6}}>
              <input value={fsText} onChange={handleFsText} onBlur={()=>setFsText(String(fs))} style={{...SEL,width:82,padding:'4px 6px'}}/>
              <span style={{fontSize:11,color:AX}}>Hz</span>
            </div>
            <input type="range" min={LOG_LO} max={LOG_HI} step={0.001} value={fsLog}
              onChange={e=>setFsFromLog(e.target.value)} style={{width:'100%',accentColor:'#f97316'}}/>
            <div style={{display:'flex',justifyContent:'space-between',fontSize:9,color:AX,marginTop:2,...MO}}>
              {['300','1k','4k','16k','44k'].map(v=><span key={v}>{v}</span>)}
            </div>
            <div style={{padding:'3px 8px',borderRadius:4,fontSize:11,marginTop:6,...MO,background:aliasingOn?'var(--color-background-danger)':'var(--color-background-success)',color:aliasingOn?'var(--color-text-danger)':'var(--color-text-success)',border:`0.5px solid ${aliasingOn?'var(--color-border-danger)':'var(--color-border-success)'}`}}>
              Nyquist: {ny.toFixed(0)} Hz {aliasingOn?'→ Aliasing!':'✓ OK'}
            </div>
          </Panel>

          <Panel title="🔉 Tiefpass (LPF)" accent="#4ade80">
            <ChkBox label="Tiefpass aktiv" checked={lpfOn} onChange={setLpfOn} color="#4ade80"/>
            {lpfOn&&<>
              <select value={lpfType} onChange={e=>setLpfType(e.target.value)} style={{...SEL,marginTop:4,marginBottom:8}}>
                {Object.entries(FILT).map(([k,v])=><option key={k} value={k}>{v}</option>)}
              </select>
              <Slider label="Grenzfrequenz" value={lpfCut} unit=" Hz"
                min={100} max={Math.max(100,Math.min(20000,Math.floor(ny*0.98)))} step={1}
                onChange={setLpfCut} accent="#4ade80"/>
              <div style={{fontSize:10,color:AX,...MO,lineHeight:1.6,marginTop:2}}>
                {lpfType==='bw4'&&'2 Biquad-Stufen in Reihe. Deutlich schärfer als BW2.'}
                {lpfType==='bw8'&&'4 Biquad-Stufen (Q: 0,51 / 0,60 / 0,90 / 2,56). Schärfster Butterworth hier.'}
                {lpfType==='ideal'&&'FFT→Maske→IFFT. Physikalisch nicht realisierbar. Gibbs-Ringing möglich.'}
                {lpfType==='rc1'&&'Einfaches RC-Glied. Kaum Wirkung auf weit entfernte ZOH-Bilder.'}
                {lpfType==='sk2'&&'Resonante Stufe — leichter Überschwinger nahe fc.'}
              </div>
            </>}
          </Panel>

          {/* Schaltungseffekte */}
          <div style={{border:'0.5px solid var(--color-border-tertiary)',borderRadius:8,overflow:'hidden',marginBottom:8}}>
            <button onClick={()=>setEfxOpen(x=>!x)} style={{width:'100%',display:'flex',justifyContent:'space-between',padding:'7px 12px',background:'var(--color-background-secondary)',color:AX,fontSize:10,border:'none',cursor:'pointer',...MO,textTransform:'uppercase',letterSpacing:'0.07em',fontWeight:500}}>
              <span>🔌 Schaltungseffekte</span><span>{efxOpen?'▲':'▼'}</span>
            </button>
            {efxOpen&&<div style={{padding:12,borderTop:'0.5px solid var(--color-border-tertiary)'}}>
              <ChkBox label="Kondensator-Droop" checked={droopOn} onChange={setDroopOn} color="#fbbf24" note="Exponentieller Spannungsabfall durch Leckstrom"/>
              {droopOn&&<Slider label="Zeitkonstante τ" value={droopTau} unit=" ms" min={1} max={1000} step={1} onChange={setDroopTau} accent="#fbbf24"/>}
              <ChkBox label="Apertur-Effekt (555-Pulsbreite)" checked={aperOn} onChange={setAperOn} color="#fbbf24" note="Endliche Sampledauer → Sinc-Rolloff"/>
              {aperOn&&<Slider label="Apertur-Anteil" value={aperPct} unit="%" min={1} max={50} step={1} onChange={setAperPct} accent="#fbbf24"/>}
            </div>}
          </div>

          <Panel title="🔊 Audio">
            <div style={{display:'flex',gap:6,marginBottom:8}}>
              {[[inSig,inRef,inPlay,setInPlay,'#60a5fa','#1d4ed8','Eingang'],[outSig,outRef,outPlay,setOutPlay,'#4ade80','#16a34a','Ausgang']].map(([sig,ref,pl,setP,col,bg,lbl])=>(
                <button key={lbl} onClick={()=>togPlay(sig,ref,pl,setP)} style={{flex:1,padding:'5px 4px',borderRadius:4,cursor:'pointer',fontSize:11,...MO,border:`0.5px solid ${pl?col:'var(--color-border-secondary)'}`,background:pl?bg+'20':'var(--color-background-secondary)',color:pl?col:AX}}>
                  {pl?'⏹':'▶'} {lbl}
                </button>
              ))}
            </div>
            <ChkBox label="S&H-Signal anzeigen" checked={showSaH} onChange={setShowSaH} color="#f97316"/>
          </Panel>
        </div>

        {/* ── Visualisierung ── */}
        <div style={{flex:1,display:'flex',flexDirection:'column',minWidth:0}}>
          <div style={{display:'flex',background:'var(--color-background-secondary)',borderBottom:'0.5px solid var(--color-border-tertiary)'}}>
            {[['time','📈 Zeitbereich'],['spec','📊 Spektrum'],['tf','〰 Übertragungsfunktion']].map(([k,l])=>(
              <TabBtn key={k} active={tab===k} onClick={()=>setTab(k)}>{l}</TabBtn>
            ))}
          </div>

          <div style={{flex:1,overflowY:'auto',padding:'16px 20px'}}>

            {/* ── ZEITBEREICH ── */}
            {tab==='time'&&(
              <div>
                <div style={{display:'flex',gap:16,marginBottom:10,flexWrap:'wrap',fontSize:12}}>
                  <span style={{color:'#60a5fa'}}>— Eingang</span>
                  {showSaH&&<span style={{color:'#f97316'}}>— S&amp;H (ZOH-Treppeninterpolation)</span>}
                  <span style={{color:'#4ade80'}}>— Filterausgang</span>
                </div>
                <div style={{background:'var(--color-background-secondary)',borderRadius:8,border:'0.5px solid var(--color-border-tertiary)',paddingTop:8}}>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={timeData} margin={{top:10,right:20,bottom:30,left:52}}>
                      <CartesianGrid strokeDasharray="2 6" stroke={GR}/>
                      <XAxis dataKey="t" stroke={AX} tick={TK13} type="number" tickFormatter={v=>v.toFixed(1)}
                        label={{value:'Zeit (ms)',position:'insideBottomRight',dy:16,fontSize:13,fill:AX}}/>
                      <YAxis stroke={AX} tick={TK13} domain={[-1.3,1.3]}
                        label={{value:'Amplitude',angle:-90,position:'insideLeft',dx:-8,fontSize:12,fill:AX}}/>
                      <Tooltip content={<TipBox fmtLabel={l=>`${parseFloat(l).toFixed(2)} ms`} fmtVal={v=>v.toFixed(4)}/>}/>
                      <Line type="linear" dataKey="Ein" stroke="#60a5fa" dot={false} strokeWidth={1.5} name="Eingang"/>
                      {showSaH&&<Line type="linear" dataKey="S&H" stroke="#f97316" dot={false} strokeWidth={1.5} name="S&H"/>}
                      <Line type="linear" dataKey="Aus" stroke="#4ade80" dot={false} strokeWidth={2} name="Ausgang"/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{marginTop:10,fontSize:11,color:AX,lineHeight:1.8,...MO}}>
                  fs={fs}Hz → T_s={(1000/fs).toFixed(3)}ms.{' '}
                  {droopOn&&`Droop τ=${droopTau}ms. `}
                  {aperOn&&`Apertur ${aperPct}%. `}
                  {lpfOn&&`Filter: ${FILT[lpfType]?.split('(')[0].trim()} bei fc=${lpfCut}Hz.`}
                  {lpfType==='ideal'&&lpfOn&&' (FFT-basiert — Gibbs-Ringing an Blockkanten möglich)'}
                </div>
              </div>
            )}

            {/* ── SPEKTRUM ── */}
            {tab==='spec'&&(
              <div>
                <div style={{display:'flex',gap:10,marginBottom:8,flexWrap:'wrap',fontSize:12}}>
                  <span style={{color:'#60a5fa'}}>— Eingang</span>
                  {showSaH&&<span style={{color:'#f97316'}}>— S&amp;H</span>}
                  <span style={{color:'#4ade80'}}>— Ausgang</span>
                  <span style={{color:'#60a5fa',marginLeft:8}}>│ Grundton/Oberton</span>
                  <span style={{color:'#f87171'}}>│ Alias (irreversibel!)</span>
                  <span style={{color:'#c084fc'}}>│ ZOH-Bild (filterbar)</span>
                  <span style={{color:'#fbbf24'}}>│ Nyquist</span>
                </div>
                <div style={{background:'var(--color-background-secondary)',borderRadius:8,border:'0.5px solid var(--color-border-tertiary)',paddingTop:8}}>
                  <ResponsiveContainer width="100%" height={340}>
                    <LineChart data={specComb} margin={{top:10,right:20,bottom:32,left:62}}>
                      <CartesianGrid strokeDasharray="2 6" stroke={GR}/>
                      <XAxis dataKey="f" stroke={AX} tick={TK13} type="number" domain={[0,MAX_F]}
                        tickFormatter={v=>v>=1000?`${(v/1000).toFixed(0)}k`:String(v)}
                        label={{value:'Frequenz (Hz)',position:'insideBottomRight',dy:18,fontSize:13,fill:AX}}/>
                      <YAxis stroke={AX} tick={TK13} domain={[-100,5]}
                        label={{value:'dBFS',angle:-90,position:'insideLeft',dx:-12,fontSize:13,fill:AX}}/>
                      <Tooltip content={<TipBox fmtLabel={l=>`${parseFloat(l).toFixed(0)} Hz`} fmtVal={v=>`${v.toFixed(1)} dBFS`}/>}/>

                      {selPeaks.map((p,i)=>(
                        <ReferenceLine key={i} x={p.f} stroke={p.color}
                          strokeDasharray={p.type==='alias'?'3 2':'4 3'} strokeOpacity={0.85} strokeWidth={1.5}
                          label={{value:p.label.slice(0,12),position:'insideTopRight',fontSize:9,fill:p.color,angle:-55,dy:-4}}/>
                      ))}
                      <ReferenceLine x={ny} stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="6 2"
                        label={{value:`Nyquist ${ny}Hz`,position:'top',fontSize:12,fill:'#fbbf24',...MO}}/>
                      {lpfOn&&<ReferenceLine x={lpfCut} stroke="#4ade80" strokeWidth={1} strokeDasharray="4 3"
                        label={{value:`fc=${lpfCut}Hz`,position:'top',fontSize:12,fill:'#4ade80',...MO}}/>}

                      <Line type="linear" dataKey="Eingang" stroke="#60a5fa" dot={false} strokeWidth={1} strokeOpacity={0.8}/>
                      {showSaH&&<Line type="linear" dataKey="S&H" stroke="#f97316" dot={false} strokeWidth={1} strokeOpacity={0.75}/>}
                      <Line type="linear" dataKey="Ausgang" stroke="#4ade80" dot={false} strokeWidth={2}/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{marginTop:10,fontSize:11,lineHeight:1.9,...MO}}>
                  <span style={{color:'var(--color-text-primary)',fontWeight:500}}>Komponenten: </span>
                  {pkList.slice(0,24).map((p,i)=>(
                    <span key={i} style={{color:p.color,marginRight:10}}>{p.f.toFixed(0)}Hz: {p.label.slice(0,14)}</span>
                  ))}
                </div>
                <div style={{marginTop:8,fontSize:11,color:AX,lineHeight:1.7,...MO}}>
                  <span style={{color:'#f87171',fontWeight:500}}>Alias (rot):</span> Harmonische &gt; Nyquist falten ins Basisband. Nicht durch LPF entfernbar.{' '}
                  <span style={{color:'#c084fc',fontWeight:500}}>ZOH-Bilder (lila):</span> Repliken bei n·fs. BW8/Ideal dämpfen diese sehr wirksam.
                </div>
              </div>
            )}

            {/* ── ÜBERTRAGUNGSFUNKTION ── */}
            {tab==='tf'&&(
              <div>
                <div style={{display:'flex',gap:14,marginBottom:8,flexWrap:'wrap',fontSize:12}}>
                  <span style={{color:'#f97316'}}>— ZOH Sinc</span>
                  <span style={{color:'#4ade80'}}>— Tiefpass ({lpfType})</span>
                  <span style={{color:'#60a5fa'}}>— Gesamtübertragung</span>
                </div>
                <div style={{background:'var(--color-background-secondary)',borderRadius:8,border:'0.5px solid var(--color-border-tertiary)',paddingTop:8}}>
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={tfData} margin={{top:10,right:20,bottom:30,left:58}}>
                      <CartesianGrid strokeDasharray="2 6" stroke={GR}/>
                      <XAxis dataKey="f" stroke={AX} tick={TK12} type="number"
                        tickFormatter={v=>v>=1000?`${(v/1000).toFixed(0)}k`:String(v)}
                        label={{value:'Frequenz (Hz)',position:'insideBottomRight',dy:14,fontSize:13,fill:AX}}/>
                      <YAxis stroke={AX} tick={TK12} domain={[-80,5]}
                        label={{value:'dB',angle:-90,position:'insideLeft',dx:-10,fontSize:13,fill:AX}}/>
                      <Tooltip content={<TipBox fmtLabel={l=>`${parseFloat(l).toFixed(0)} Hz`} fmtVal={v=>`${v.toFixed(1)} dB`}/>}/>
                      <ReferenceLine y={-3} stroke={AX} strokeDasharray="4 2"
                        label={{value:'-3 dB',position:'insideTopLeft',fontSize:12,fill:AX,...MO}}/>
                      <ReferenceLine y={-3.92} stroke="#f97316" strokeDasharray="2 4" strokeOpacity={0.5}
                        label={{value:'ZOH@Nyquist: −3,92dB',position:'insideBottomLeft',fontSize:10,fill:'#f97316',...MO}}/>
                      <ReferenceLine x={ny} stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="6 2"
                        label={{value:'Nyquist',position:'top',fontSize:12,fill:'#fbbf24',...MO}}/>
                      <ReferenceLine x={fs} stroke={AX} strokeDasharray="3 3"
                        label={{value:'fs',position:'top',fontSize:12,fill:AX,...MO}}/>
                      {lpfOn&&<ReferenceLine x={lpfCut} stroke="#4ade80" strokeDasharray="4 3"
                        label={{value:'fc',position:'top',fontSize:12,fill:'#4ade80',...MO}}/>}
                      <ReferenceLine x={freq} stroke="#60a5fa" strokeDasharray="2 4"
                        label={{value:'f₀',position:'top',fontSize:12,fill:'#60a5fa',...MO}}/>
                      <Line type="linear" dataKey="ZOH"    stroke="#f97316" dot={false} strokeWidth={1.5} name="ZOH Sinc"/>
                      <Line type="linear" dataKey="LPF"    stroke="#4ade80" dot={false} strokeWidth={1.5} name="Tiefpass"/>
                      <Line type="linear" dataKey="Gesamt" stroke="#60a5fa" dot={false} strokeWidth={2}   name="Gesamt"/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{marginTop:12,fontSize:11,color:AX,lineHeight:1.8,...MO}}>
                  ZOH-Sinc: −3,92 dB bei Nyquist, Nullstellen bei n·fs.{' '}
                  BW8 hat −48 dB/Okt = 16-fache Dämpfung pro Oktave.{' '}
                  Idealfilter springt auf −∞ bei fc (nicht analogrealisierbar, aber als Referenz nützlich).
                </div>
              </div>
            )}
          </div>

          {/* Statusleiste */}
          <div style={{background:'var(--color-background-secondary)',borderTop:'0.5px solid var(--color-border-tertiary)',padding:'4px 14px',display:'flex',flexWrap:'wrap',gap:14,fontSize:10,color:AX,...MO}}>
            <span style={{color:'#f97316'}}>fs={fs}Hz</span>
            <span>Nyquist={ny}Hz</span>
            <span style={{color:'#60a5fa'}}>f₀={freq}Hz</span>
            <span>T_s={(1000/fs).toFixed(3)}ms</span>
            {lpfOn&&<span style={{color:'#4ade80'}}>fc={lpfCut}Hz ({lpfType})</span>}
            {droopOn&&<span style={{color:'#fbbf24'}}>τ={droopTau}ms</span>}
            {aperOn&&<span style={{color:'#fbbf24'}}>Aper={aperPct}%</span>}
            {aliasingOn&&<span style={{color:'var(--color-text-danger)'}}>⚠ ALIASING!</span>}
          </div>
        </div>
      </div>
    </div>
  );
}