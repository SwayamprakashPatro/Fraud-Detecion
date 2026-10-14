/* GraphSAGE Fraud Explorer: app logic.
   Data comes from js/data.js (window.GML_DATA) or from a file the user drops in.
   Sections: constants, data, stats, home, network, other pictures, import, wiring. */
(function(){
"use strict";
var P=null;   /* set by boot(): the live API payload, or the offline snapshot from js/data.js */

/* ---------- constants ---------- */
var TY={
  "Payment Fraud":["--t1","Unauthorised or deceptive card and bank payments."],
  "UPI Fraud":["--t2","Scams that use UPI requests, QR codes or fake handles."],
  "Account Takeover":["--t3","Someone else got into an account and spent its money."],
  "Money Laundering":["--t4","Money passed through accounts to hide where it came from."],
  "Mule Account":["--t5","An account that receives stolen money and passes it along."],
  "Identity Fraud":["--t6","Someone used another person's identity details."]
};
var EXTRA=["--t1","--t2","--t3","--t4","--t5","--t6"];
var OUT={TP:["Fraud caught","--tp"],FN:["Fraud missed","--fn"],FP:["Normal payment flagged by mistake","--fp"],TN:["Normal, passed correctly","--tn"],TR:["Not part of the held-out test","--legit"]};
var NONE="No fraud",KIND={A:"Account",D:"Device",M:"Shop",U:"UPI ID"};
var $=function(id){return document.getElementById(id);};

/* ---------- state ---------- */
var D=null,ND={},IDX={},S={mode:"label",view:"net",types:new Set(),sel:null,hot:new Set(),th:.28,links:false,sampled:0};
var cache=new Map(),sim=null,svg=null,gRoot=null,zoom=null;
var linkSel=null,nodeSel=null,haloSel=null,adj=new Map(),fitT=0,lastW=0,lastH=0,raf=0,homeAnimated=false;

/* ---------- small helpers ---------- */
function css(v){return "var("+v+")";}
function hash(s){var h=0;for(var i=0;i<s.length;i++)h=(h*31+s.charCodeAt(i))|0;return h;}
function sum(a){return a.reduce(function(x,y){return x+y;},0);}
function hh(h){return String(h).padStart(2,"0");}
function eid(x){return typeof x==="object"?x.id:x;}
function inr(v){return "₹"+Math.round(v).toLocaleString("en-IN");}
function pct(x,d){return isFinite(x)?(x*100).toFixed(d==null?1:d)+"%":"n/a";}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c];});}
function typesList(){var s=new Set();D.txns.forEach(function(t){if(t.fraud)s.add(t.fraud);});return Array.from(s).sort();}
function typeColor(ty){if(!ty)return css("--legit");if(TY[ty])return css(TY[ty][0]);return css(EXTRA[Math.abs(hash(ty))%EXTRA.length]);}
function kind(n){return KIND[n.type]||"Node";}
function fmtDate(ts){if(!ts)return "";var d=new Date(ts.slice(0,10)+"T00:00:00");return isNaN(d)?ts:d.toLocaleDateString("en-IN",{day:"numeric",month:"short",year:"numeric"});}
function flag(t){return t.pr!=null&&t.pr>=S.th;}
function outcome(t){if(!D.hasPred||!t.test)return "TR";var f=flag(t);return t.fraud?(f?"TP":"FN"):(f?"FP":"TN");}

/* ---------- data ---------- */
function fromProject(){
  var ix={};P.cols.forEach(function(k,i){ix[k]=i;});
  var tx=P.rows.map(function(r){var ts=r[ix.ts];
    return {id:String(r[ix.id]),s:r[ix.s],r:r[ix.r],dev:r[ix.dev],m:r[ix.m],u:r[ix.u],amt:r[ix.amt],ch:r[ix.ch],ts:ts,hr:parseInt(ts.slice(11,13),10)||0,fraud:r[ix.ft],pr:r[ix.p],sp:r[ix.sp],test:r[ix.sp]===2};});
  return {txns:tx,hasPred:true,project:true,th0:P.threshold,fixed:false};
}
function setData(data){
  D=data;ND={};IDX={};
  var add=function(id,t){if(id==null)return;if(!ND[id])ND[id]={id:id,type:id.charAt(0)};(IDX[id]=IDX[id]||[]).push(t);};
  D.txns.forEach(function(t){add(t.s,t);add(t.r,t);add(t.dev,t);add(t.m,t);add(t.u,t);});
  D.base=D.txns.filter(function(t){return t.fraud;}).length/D.txns.length;
  D.T=D.hasPred?D.txns.filter(function(t){return t.test&&t.pr!=null;}):[];
  if(D.hasPred&&!D.T.length)D.hasPred=false;
  if(D.hasPred)curves();
  cache.clear();S.sel=null;S.mode="label";S.links=false;S.types=new Set(typesList());S.th=D.th0!=null?D.th0:.5;
  $("f-links").checked=false;
  var b=$("banner");b.className="pill"+(D.project&&!P.live?"":" live");
  b.innerHTML="<i></i>"+(D.project?(P.live?"Live from the model API":"Snapshot, no server")+" · GraphSAGE · "+D.txns.length.toLocaleString("en-IN")+" payments":"Your data: "+D.txns.length.toLocaleString("en-IN")+" payments");
  $("demo-btn").hidden=!!D.project;$("reload").hidden=!(D.project&&P.live);
  document.querySelectorAll(".project-only").forEach(function(e){e.hidden=!D.project;});
  buildKinds();buildHot();syncControls();drawHome();
  if(!$("v-net").hidden)drawNet(true);
}

/* ---------- node statistics and plain-language roles ---------- */
function nstat(id){
  var L=IDX[id]||[],o={n:L.length,fr:0,sent:0,recv:0,val:0,by:{},tn:0,caught:0,missed:0,fa:0,accts:new Set(),devs:new Set(),shops:new Set(),upis:new Set()};
  L.forEach(function(t){
    if(t.fraud){o.fr++;o.val+=t.amt;o.by[t.fraud]=(o.by[t.fraud]||0)+1;if(t.s===id)o.sent++;else if(t.r===id)o.recv++;}
    o.accts.add(t.s);if(t.dev)o.devs.add(t.dev);if(t.m)o.shops.add(t.m);if(t.u)o.upis.add(t.u);
    if(D.hasPred&&t.test&&t.pr!=null){var f=flag(t);o.tn++;if(t.fraud){if(f)o.caught++;else o.missed++;}else if(f)o.fa++;}
  });
  return o;
}
function role(n,o){
  var base=D.base||.2,rate=o.n?o.fr/o.n:0,hi=o.n>=3&&rate>=1.5*base;
  if(!o.fr)return ["No fraud seen","None of its "+o.n+" payment"+(o.n===1?" is":"s is")+" fraud.","ok"];
  if(n.type==="A"){
    if(o.recv>=3&&o.recv>=2*o.sent)return ["Collection point","Fraud keeps arriving here: "+o.recv+" payments in, "+o.sent+" out. Scammers often pool stolen money in a few accounts like this.","bad"];
    if(o.sent>=3&&o.sent>=2*o.recv)return ["Possible victim","Fraud keeps leaving this account: "+o.sent+" payments out, "+o.recv+" in. It may have been taken over or targeted.","warn"];
  }
  if(hi){var w={A:"This account",D:"This device",M:"This shop",U:"This UPI ID"}[n.type];return ["High fraud share",w+" is tied to "+pct(rate,0)+" fraud, against "+pct(base,0)+" overall"+(n.type!=="A"&&o.accts.size>1?", across "+o.accts.size+" accounts":"")+".","bad"];}
  if(rate<=.5*base)return ["Mostly normal","Some fraud is linked here, but it is a small share ("+pct(rate,0)+") of its payments.","ok"];
  return ["Touched by fraud","Fraud passes through here: "+pct(rate,0)+" of its payments, close to the overall rate of "+pct(base,0)+".","warn"];
}

/* ---------- metrics and curves ---------- */
function metrics(){
  var tp=0,fp=0,fn=0,tn=0;
  D.T.forEach(function(t){var o=outcome(t);if(o==="TP")tp++;else if(o==="FP")fp++;else if(o==="FN")fn++;else tn++;});
  var n=D.T.length,pos=tp+fn,neg=tn+fp,prec=tp?tp/(tp+fp):0,rec=pos?tp/pos:0;
  return {n:n,tp:tp,fp:fp,fn:fn,tn:tn,pos:pos,neg:neg,prec:prec,rec:rec,f1:prec+rec?2*prec*rec/(prec+rec):0,fpr:neg?fp/neg:0,acc:(tp+tn)/n,naive:neg/n};
}
function curves(){
  var T=D.T,pos=T.filter(function(t){return t.fraud;}).length,neg=T.length-pos;
  var s=T.slice().sort(function(a,b){return b.pr-a.pr;}),tp=0,fp=0,pt=0,pf=0,roc=[[0,0]],prc=[],auc=0,ap=0;
  if(!pos||!neg){D.roc=D.prc=null;D.auc=D.ap=NaN;return;}
  for(var i=0;i<s.length;i++){
    if(s[i].fraud)tp++;else fp++;
    if(i===s.length-1||s[i+1].pr!==s[i].pr){
      roc.push([fp/neg,tp/pos]);prc.push([tp/pos,tp/(tp+fp)]);
      auc+=(fp-pf)/neg*(tp+pt)/pos/2;ap+=(tp-pt)/pos*(tp/(tp+fp));pt=tp;pf=fp;
    }
  }
  D.roc=roc;D.prc=prc;D.auc=auc;D.ap=ap;
}

/* ---------- Home ---------- */
function ring(v,color,label,sub,anim){
  var c=2*Math.PI*42,o=c*(1-(isFinite(v)?v:0));
  return '<div class="ring"><svg viewBox="0 0 100 100" width="108" height="108" role="img" aria-label="'+label+' '+pct(v)+'"><circle cx="50" cy="50" r="42" fill="none" stroke="var(--surface2)" stroke-width="9"/><circle cx="50" cy="50" r="42" fill="none" stroke="'+color+'" stroke-width="9" stroke-linecap="round" class="fg" stroke-dasharray="'+c+'" style="stroke-dashoffset:'+(anim?c:o)+'" data-o="'+o+'" transform="rotate(-90 50 50)"/><text x="50" y="56" text-anchor="middle" class="rv">'+(isFinite(v)?(v>=.99&&v<1?(v*100).toFixed(1):Math.round(v*100))+"%":"n/a")+'</text></svg><b>'+label+'</b><span>'+sub+'</span></div>';
}
function animateRings(){requestAnimationFrame(function(){requestAnimationFrame(function(){document.querySelectorAll("#rings .fg").forEach(function(c){c.style.strokeDashoffset=c.getAttribute("data-o");});});});}
function tiles(K){$("tiles").innerHTML=K.map(function(k){return '<div class="card tile"><div class="v">'+k[0]+'</div><div class="l">'+k[1]+'</div><div class="s">'+k[2]+'</div></div>';}).join("");}
function drawHome(){
  var anim=!homeAnimated;homeAnimated=true;
  var th=$("th");th.value=S.th;th.disabled=!!D.fixed||!D.hasPred;
  $("thv").textContent=S.th.toFixed(2);
  if(!D.hasPred){noPred();return;}
  var m=metrics();
  $("eyebrow").textContent=D.project?"Results on held-out payments":"Results on your held-out payments";
  $("headline").textContent="The model caught "+Math.round(m.rec*100)+"% of the fraud in "+m.n.toLocaleString("en-IN")+" payments it never trained on.";
  $("headsub").textContent="It raised "+m.fp+" false alarm"+(m.fp===1?"":"s")+" among "+m.neg.toLocaleString("en-IN")+" normal payments and missed "+m.fn+" of "+m.pos+" fraud payments. Alerts fire when the fraud score is "+S.th.toFixed(2)+" or higher.";
  $("rings").innerHTML=ring(m.rec,css("--accent"),"Fraud caught","of real fraud spotted",anim)+ring(m.prec,css("--t4"),"Alerts that were real","of alerts were fraud",anim)+ring(D.auc,css("--t3"),"Ranking quality","ROC-AUC: fraud scored above normal",anim);
  if(anim)animateRings();
  tiles([[pct(m.fpr,2),"False alarms","of normal payments were flagged"],[pct(m.acc),"Overall correct","vs "+pct(m.naive)+" if it flagged nothing"],[pct(m.f1),"F1 score","balance of fraud caught and alert accuracy"]]);
  drawExplorer(m);drawROC(m);drawPRC(m);drawByType();
  if(D.project){drawHist();drawFacts();drawReality();}else $("prov").textContent="";
}
function noPred(){
  var n=D.txns.length,fr=D.txns.filter(function(t){return t.fraud;}),val=sum(fr.map(function(t){return t.amt;})),rate=fr.length/n;
  $("eyebrow").textContent="Your file";
  $("headline").textContent="Your file has "+n.toLocaleString("en-IN")+" payments and "+fr.length.toLocaleString("en-IN")+" of them are fraud.";
  $("headsub").textContent="No usable model scores were found, so accuracy is not shown. Add a Predicted_Prob column and a Split column (or Is_Test) to see how well a model performs.";
  $("rings").innerHTML=ring(rate,css("--tp"),"Fraud share","of all payments",false);
  tiles([[n.toLocaleString("en-IN"),"Payments","in the file"],[fr.length.toLocaleString("en-IN"),"Fraud payments",pct(rate)+" of all payments"],[inr(val),"Fraud value","total of fraud payments"]]);
  var msg='<p class="note" style="margin:0">Needs a Predicted_Prob column.</p>';
  ["cm1","roc","prc","bytype","trade","mrow"].forEach(function(id){var e=$(id);if(e.tagName.toLowerCase()==="svg")e.innerHTML="";else e.innerHTML=id==="cm1"||id==="bytype"?msg:"";});
  ["cm1n","rocn","prn","bytypen","thnote","histn"].forEach(function(id){$(id).textContent="";});
}
function drawExplorer(m){
  var cell=function(v,c){return 'style="background:color-mix(in srgb,'+c+' '+Math.round(8+70*v/(Math.max(m.tp,m.fp,m.fn,m.tn)||1))+'%,var(--surface))"';};
  $("cm1").innerHTML='<div class="cm" style="grid-template-columns:90px 1fr 1fr"><div class="h"></div><div class="h">Said normal</div><div class="h">Said fraud</div>'+
    '<div class="h">Was normal</div><div class="mono" '+cell(m.tn,"var(--tn)")+'>'+m.tn.toLocaleString("en-IN")+'</div><div class="mono" '+cell(m.fp,"var(--fp)")+'>'+m.fp+'</div>'+
    '<div class="h">Was fraud</div><div class="mono" '+cell(m.fn,"var(--fn)")+'>'+m.fn+'</div><div class="mono" '+cell(m.tp,"var(--tp)")+'>'+m.tp+'</div></div>';
  $("mrow").innerHTML=[["Precision",m.prec],["Recall",m.rec],["F1",m.f1],["Accuracy",m.acc]].map(function(k){return '<div><span>'+k[0]+'</span><b class="mono">'+pct(k[1])+'</b></div>';}).join("");
  $("trade").innerHTML='<div><b class="mono">'+m.fn+'</b>fraud payments missed, out of '+m.pos+'</div><div><b class="mono">'+m.fp+'</b>normal payments wrongly flagged, out of '+m.neg.toLocaleString("en-IN")+'</div>';
  $("cm1n").textContent="Held-out payments only ("+m.n.toLocaleString("en-IN")+"). Flagging nothing would already be "+pct(m.naive)+" accurate, so accuracy alone says little.";
  var saved=D.th0!=null&&Math.abs(S.th-D.th0)<.005;
  $("thnote").textContent=D.fixed?"Your file has yes or no predictions, so there is no threshold to move.":
    (D.project?"The project picked "+D.th0.toFixed(2)+" on the validation set to get the best F1. ":"")+"Lower it to catch more fraud at the cost of more false alarms. Raise it to bother fewer customers and miss more fraud."+(saved||D.th0==null?"":" You are exploring: the saved value is "+D.th0.toFixed(2)+".");
  $("th-reset").hidden=D.th0==null||saved;
}
/* chart helpers */
function frame(id,xl,yl){
  var g='<g class="ax">';
  [0,.25,.5,.75,1].forEach(function(v){var x=40+v*240,y=220-v*204;g+='<line x1="40" x2="280" y1="'+y+'" y2="'+y+'" stroke="var(--line)"/><line y1="16" y2="220" x1="'+x+'" x2="'+x+'" stroke="var(--line)" stroke-opacity=".5"/>';});
  g+='</g>';
  [0,.5,1].forEach(function(v){g+='<text x="'+(40+v*240)+'" y="236" text-anchor="middle" class="mono">'+v.toFixed(1)+'</text><text x="34" y="'+(224-v*204)+'" text-anchor="end" class="mono">'+v.toFixed(1)+'</text>';});
  g+='<text x="160" y="262" text-anchor="middle">'+xl+'</text><text transform="translate(9 118) rotate(-90)" text-anchor="middle">'+yl+'</text>';
  return g;
}
var CX=function(v){return 40+v*240;},CY=function(v){return 220-v*204;};
function drawROC(m){
  if(!D.roc){$("roc").innerHTML="";$("rocn").textContent="Needs both fraud and normal payments in the test set.";return;}
  var d=D.roc.map(function(p,i){return (i?"L":"M")+CX(p[0]).toFixed(1)+" "+CY(p[1]).toFixed(1);}).join("");
  $("roc").innerHTML=frame("roc","False alarms (share of normal payments flagged)","Fraud caught")+
    '<line x1="'+CX(0)+'" y1="'+CY(0)+'" x2="'+CX(1)+'" y2="'+CY(1)+'" stroke="var(--muted)" stroke-dasharray="4 4" stroke-opacity=".6"/>'+
    '<path d="'+d+'L'+CX(1)+" "+CY(0)+'Z" fill="var(--accent)" fill-opacity=".1"/><path d="'+d+'" fill="none" stroke="var(--accent)" stroke-width="2.4" stroke-linejoin="round"/>'+
    '<circle cx="'+CX(m.fpr)+'" cy="'+CY(m.rec)+'" r="5.5" fill="var(--tp)" stroke="var(--surface)" stroke-width="2"/>'+
    '<text x="276" y="202" text-anchor="end" class="tt">AUC '+D.auc.toFixed(4)+'</text><text x="276" y="216" text-anchor="end">dashed: pure guessing</text>';
  $("rocn").textContent="Each point is one alert threshold. The red dot is the threshold you have set: "+pct(m.rec)+" of fraud caught for "+pct(m.fpr,2)+" false alarms.";
}
function drawPRC(m){
  if(!D.prc){$("prc").innerHTML="";$("prn").textContent="Needs fraud payments in the test set.";return;}
  var d=D.prc.map(function(p,i){return (i?"L":"M")+CX(p[0]).toFixed(1)+" "+CY(p[1]).toFixed(1);}).join("");
  var base=m.pos/m.n;
  $("prc").innerHTML=frame("prc","Recall (share of fraud caught)","Precision (share of alerts that were fraud)")+
    '<line x1="'+CX(0)+'" y1="'+CY(base)+'" x2="'+CX(1)+'" y2="'+CY(base)+'" stroke="var(--muted)" stroke-dasharray="4 4" stroke-opacity=".6"/>'+
    '<path d="'+d+'" fill="none" stroke="var(--t4)" stroke-width="2.4" stroke-linejoin="round"/>'+
    '<circle cx="'+CX(m.rec)+'" cy="'+CY(m.prec)+'" r="5.5" fill="var(--tp)" stroke="var(--surface)" stroke-width="2"/>'+
    '<text x="276" y="'+(CY(base)-5)+'" text-anchor="end">dashed: guessing ('+pct(base,0)+' are fraud)</text><text x="40" y="12" class="tt">Average precision '+D.ap.toFixed(4)+'</text>';
  $("prn").textContent="Shows what each extra bit of fraud caught costs in false alarms. The red dot is your current threshold.";
}
function drawHist(){
  var H=P.history,n=H.train_loss.length,X=function(i){return 40+i/(n-1)*240;};
  var mxl=Math.max(Math.max.apply(null,H.train_loss),Math.max.apply(null,H.val_loss)),YL=function(v){return 108-v/mxl*72;},YM=function(v){return 232-v*86;};
  var path=function(a,Y){return a.map(function(v,i){return (i?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1);}).join("");};
  var be=P.bestEpoch-1,g='<g class="ax"><line x1="40" x2="280" y1="108" y2="108" stroke="var(--line)"/><line x1="40" x2="280" y1="232" y2="232" stroke="var(--line)"/><line x1="40" x2="280" y1="136" y2="136" stroke="var(--line)" stroke-opacity=".5"/></g>';
  g+='<text x="40" y="12" class="tt">Loss, lower is better <tspan dx="8" style="fill:var(--tp)">&#9632; training</tspan><tspan dx="6" style="fill:var(--accent)">&#9632; validation</tspan></text><text x="40" y="130" class="tt">Validation, higher is better <tspan dx="8" style="fill:var(--t3)">&#9632; ROC-AUC</tspan><tspan dx="6" style="fill:var(--t4)">&#9632; F1</tspan></text>';
  g+='<path d="'+path(H.train_loss,YL)+'" fill="none" stroke="var(--tp)" stroke-width="2.2"/><path d="'+path(H.val_loss,YL)+'" fill="none" stroke="var(--accent)" stroke-width="2.2"/>';
  g+='<path d="'+path(H.val_roc_auc,YM)+'" fill="none" stroke="var(--t3)" stroke-width="2.2"/><path d="'+path(H.val_f1,YM)+'" fill="none" stroke="var(--t4)" stroke-width="2.2"/>';
  g+='<line x1="'+X(be)+'" x2="'+X(be)+'" y1="16" y2="232" stroke="var(--ink)" stroke-dasharray="3 3" stroke-opacity=".6"/><text x="'+(X(be)-4)+'" y="52" text-anchor="end">saved: epoch '+P.bestEpoch+'</text>';
  g+='<text x="40" y="248" class="mono">1</text><text x="280" y="248" text-anchor="end" class="mono">'+n+'</text><text x="160" y="262" text-anchor="middle">Training round (epoch)</text>';
  
  $("hist").innerHTML=g;
  var z=0;while(z<n&&H.val_f1[z]===0)z++;
  $("histn").textContent="Validation F1 stays at 0 for the first "+z+" rounds, then climbs. The saved model is the best validation ROC-AUC, from epoch "+P.bestEpoch+" of "+n+". Training and validation loss stay close, so there is no sign of heavy overfitting.";
}
function drawByType(){
  var by={};
  D.T.forEach(function(t){if(!t.fraud)return;var o=by[t.fraud]||(by[t.fraud]={n:0,c:0});o.n++;if(flag(t))o.c++;});
  var ks=Object.keys(by).sort(function(a,b){return by[b].c/by[b].n-by[a].c/by[a].n||by[b].n-by[a].n;});
  if(!ks.length){$("bytype").innerHTML='<p class="note" style="margin:0">No fraud in the held-out payments.</p>';$("bytypen").textContent="";return;}
  $("bytype").innerHTML=ks.map(function(k){var o=by[k],r=o.c/o.n;return '<div class="tbar"><span title="'+esc(k)+'">'+esc(k)+'</span><span class="t"><i style="width:'+(r*100)+'%;background:'+typeColor(k)+'"></i></span><span class="mono">'+o.c+' of '+o.n+'</span></div>';}).join("");
  $("bytypen").textContent="Share of held-out fraud the model flagged, per kind. The model only says fraud or not fraud. It never names the kind.";
}
function drawFacts(){
  var F=P.facts,ts=D.txns.map(function(t){return t.ts;}).filter(Boolean).sort(),mo=function(s){return new Date(s.slice(0,10)+"T00:00:00").toLocaleDateString("en-IN",{month:"short",year:"numeric"});};
  var nt=F.nodeTypes,sp=F.split,tot=sp.train+sp.val+sp.test,nf=function(v){return Number(v).toLocaleString("en-IN");};
  var f=[["Payments",nf(D.txns.length)],["Fraud",nf(sum(D.txns.map(function(t){return t.fraud?1:0;})))+" ("+pct(D.base)+")"],
    ["Network dots",nf(F.nodes)+": "+nf(nt.account)+" accounts, "+nf(nt.device)+" devices, "+nf(nt.upi)+" UPI IDs, "+nf(nt.merchant)+" shops"],["Network links",nf(F.links)],
    ["Split",Math.round(sp.train/tot*100)+"% train, "+Math.round(sp.val/tot*100)+"% validation, "+Math.round(sp.test/tot*100)+"% test"],["Model size",nf(F.params)+" trainable numbers"]];
  if(ts.length)f.splice(2,0,["Period",mo(ts[0])+" to "+mo(ts[ts.length-1])]);
  $("facts").innerHTML=f.map(function(k){return '<div><dt>'+k[0]+'</dt><dd'+(k[1].length>20?' style="font-size:14px;font-weight:600"':"")+'>'+k[1]+'</dd></div>';}).join("");
  var A=Object.assign({params:nf(F.params)},F.arch);
  document.querySelectorAll("[data-f]").forEach(function(e){e.textContent=A[e.getAttribute("data-f")];});
  var S0=P.source,ok=S0.verified===true?"They match evaluation_report.json exactly.":S0.verified===false?"They do NOT match evaluation_report.json, so check the repo files.":"",v=ok?" "+ok:"";
  $("prov").innerHTML=P.live?"Live: these numbers were just computed by the model in your repo through the API (checkpoint epoch "+P.bestEpoch+")."+v+' <a href="docs" target="_blank" rel="noopener">API docs</a>':"Snapshot of the model's output taken "+S0.generated+" (checkpoint epoch "+P.bestEpoch+")."+v+" Run python run_app.py for live numbers.";
}
function drawReality(){
  var r=P.report.test_metrics,R=P.reality,C=P.claims||{},F=P.facts,nf=function(v){return Number(v).toLocaleString("en-IN");};
  var rows=[["GraphSAGE, as saved",r.roc_auc,r.f1,r.recall,r.precision,1],["Same model, graph summaries set to zero",R.zeroed.auc,R.zeroed.f1,R.zeroed.rec,R.zeroed.prec],["Logistic regression, no graph",R.logreg.auc,R.logreg.f1,R.logreg.rec,R.logreg.prec],["Gradient boosting, no graph",R.histgb.auc,R.histgb.f1,R.histgb.rec,R.histgb.prec]];
  var best=[1,2,3,4].map(function(c){return Math.max.apply(null,rows.map(function(x){return x[c];}));});
  $("cmp").innerHTML='<tr><th>Model ('+nf(P.facts.split.test)+' test payments)</th><th>ROC-AUC</th><th>F1</th><th>Recall</th><th>Precision</th></tr>'+rows.map(function(x){return '<tr'+(x[5]?' class="me"':'')+'><td>'+x[0]+'</td>'+[1,2,3,4].map(function(c){return '<td class="mono'+(x[c]===best[c-1]?' best':'')+'">'+(x[c]*100).toFixed(2)+'%</td>';}).join("")+'</tr>';}).join("");
  $("cmpnote").textContent="Same "+nf(F.split.test)+" test payments. The last two rows use only the "+F.featureCount+" payment details, no graph, with the alert threshold chosen on the validation set. They are computed by tools/export_data.py, not taken from the repo.";
  var items=[],near=function(a,b){return Math.abs(a-b)<.0005;};
  var cl=[["F1",C.f1,r.f1],["ROC-AUC",C.auc,r.roc_auc],["PR-AUC",C.pr,r.pr_auc]].filter(function(x){return x[1]!=null;});
  var off=cl.filter(function(x){return !near(x[1],x[2]);});
  if(off.length)items.push(["The report's headline numbers do not match the saved results.","final_report.md says "+cl.map(function(x){return x[0]+" "+pct(x[1],2);}).join(", ")+". The saved evaluation file and the model itself give "+cl.map(function(x){return pct(x[2],2);}).join(", ")+". This page uses the saved results."+(C.epoch&&C.epoch!==P.bestEpoch?" The report also names epoch "+C.epoch+" as best, but the checkpoint is from epoch "+P.bestEpoch+".":"")]);
  else if(cl.length)items.push(["The report's numbers match the saved results.","final_report.md and the saved evaluation file agree on "+cl.map(function(x){return x[0]+" "+pct(x[2],2);}).join(", ")+"."]);
  var rival=Math.max(R.logreg.f1,R.histgb.f1),noLift=R.zeroed.f1>=r.f1-.01;
  if(rival>r.f1&&noLift)items.push(["The graph adds no measurable lift.","Zeroing the graph summaries in the saved model leaves F1 at "+pct(R.zeroed.f1,1)+" (saved: "+pct(r.f1,1)+"), and simple models that never see the graph score higher (best F1 "+pct(rival,1)+"). Caveat: the zeroing test is not a retrained ablation, so treat it as a strong hint, not proof."]);
  else if(rival>r.f1)items.push(["Simple models without the graph score higher.","The best graph-free model reaches F1 "+pct(rival,1)+" against "+pct(r.f1,1)+" for the saved model, though zeroing the graph summaries does cost the saved model some accuracy (F1 "+pct(R.zeroed.f1,1)+")."]);
  else items.push(["The graph model leads the graph-free baselines.","Saved F1 "+pct(r.f1,1)+" against "+pct(rival,1)+" for the best graph-free model, and zeroing the graph summaries drops it to "+pct(R.zeroed.f1,1)+"."]);
  var best_auc=Math.max(R.logreg.auc,R.histgb.auc);
  items.push(["How hard is this data?","A plain linear model reaches "+pct(R.logreg.auc,1)+" ROC-AUC and "+pct(D.base,0)+" of payments are fraud, far above real-world rates. "+(best_auc>=.99?"Scores this high from simple models suggest the data is easy and probably generated, so read them as proof that the pipeline works, not as a forecast for live traffic.":"Fraud rates this high are unusual, so be careful extending these scores to live traffic.")]);
  items.push(["The test is gentler than a live deployment.","The split is random on one shared network: the same accounts and devices appear in training and test, and the network facts use all "+nf(F.payments)+" payments, test ones included."+(F.idFeatures.length?" Account ID numbers ("+F.idFeatures.join(", ")+") are also among the "+F.featureCount+" inputs. Dropping them moved a boosted-tree baseline from "+pct(R.histgb.auc,2)+" to "+pct(R.histgbNoId.auc,2)+" ROC-AUC, so they are not driving the result here, but they should not be inputs.":"")]);
  $("finds").innerHTML=items.map(function(x,i){return '<li><i>'+(i+1)+'</i><div><b>'+x[0]+'</b> '+x[1]+'</div></li>';}).join("");
}

/* ---------- Network: controls, side panels, tooltips ---------- */
function buildKinds(){
  var box=$("f-types");box.innerHTML="";
  typesList().forEach(function(ty){
    var b=document.createElement("button");b.type="button";b.className="kind";b.dataset.ty=ty;
    b.innerHTML='<span class="sw" style="background:'+typeColor(ty)+'"></span><span><b></b><span class="d"></span></span>';
    b.querySelector("b").textContent=ty;b.querySelector(".d").textContent=TY[ty]?TY[ty][1]:"";
    b.addEventListener("click",function(){if(S.types.has(ty))S.types.delete(ty);else S.types.add(ty);syncControls();drawNet(true);});
    box.appendChild(b);
  });
}
function syncControls(){
  document.querySelectorAll("#f-types .kind").forEach(function(b){b.setAttribute("aria-pressed",S.types.has(b.dataset.ty)?"true":"false");});
  var all=S.types.has(NONE);
  $("v-all").setAttribute("aria-pressed",all?"true":"false");$("v-fraud").setAttribute("aria-pressed",all?"false":"true");
  if(!D.hasPred&&S.mode==="outcome")S.mode="label";
  $("f-miss").disabled=!D.hasPred;$("f-miss").checked=S.mode==="outcome";
  $("f-links").checked=S.links;
}
function buildHot(){
  var list=Object.keys(IDX).map(function(id){var o=nstat(id);return {id:id,n:o.fr,amt:o.val};}).filter(function(o){return o.n>0;})
    .sort(function(a,b){return b.n-a.n||b.amt-a.amt;}).slice(0,5);
  S.hot=new Set(list.map(function(o){return o.id;}));
  var el=$("hot");
  el.innerHTML='<h3>Fraud hotspots</h3><p class="note" style="margin:0 0 6px">The dots that the most fraud passes through.</p>';
  list.forEach(function(o,i){
    var b=document.createElement("button");b.type="button";b.className="hotbtn";
    b.innerHTML='<span class="rank">'+(i+1)+'</span><span><b></b><span class="d"></span></span>';
    b.querySelector("b").textContent=kind(ND[o.id])+" "+o.id;
    b.querySelector(".d").textContent=o.n+" fraud payments, "+inr(o.amt);
    b.addEventListener("click",function(){S.sel=o.id;typesList().forEach(function(ty){S.types.add(ty);});if(ND[o.id].type!=="A")S.links=true;syncControls();drawNet(true);});
    el.appendChild(b);
  });
}
function visible(){
  var v=D.txns.filter(function(t){return S.types.has(t.fraud||NONE);}),CAP=4000;S.sampled=0;
  if(v.length>CAP){
    S.sampled=v.length;
    var f=v.filter(function(t){return t.fraud;}),n=v.filter(function(t){return !t.fraud;}),keep=CAP-f.length;
    if(keep<=0)return f.slice(0,CAP);
    var step=Math.ceil(n.length/keep);
    return f.concat(n.filter(function(t,i){return i%step===0;}));
  }
  return v;
}
function legend(){
  var sh='<span style="flex-basis:100%;color:var(--ink)">Circle account'+(S.links?', square device, diamond shop, triangle UPI ID':'')+'. Line colour:</span>';
  var h=S.mode==="label"?typesList().concat([NONE]).map(function(ty){return '<span><i class="sw" style="background:'+(ty===NONE?css("--legit"):typeColor(ty))+'"></i>'+(ty===NONE?"Normal payment":ty)+'</span>';}).join("")
    :Object.keys(OUT).map(function(k){return '<span><i class="sw" style="background:'+css(OUT[k][1])+'"></i>'+OUT[k][0]+'</span>';}).join("");
  $("legend").innerHTML=sh+h;
}
function drawSide(vt){
  var el=$("side");
  if(!S.sel){
    var f=vt.filter(function(t){return t.fraud;}).length;
    el.innerHTML='<h3>Now showing</h3><div class="tile" style="padding:0"><div class="v">'+vt.length.toLocaleString("en-IN")+'</div><div class="l">payments</div><div class="s">'+f.toLocaleString("en-IN")+' of them are fraud ('+pct(f/(vt.length||1))+')</div></div><div class="callout">Hover any dot for its story. Click a dot, or a hotspot below, to focus on who it is connected to.</div>'+(S.sampled?'<div class="callout">Showing a sample: '+vt.length.toLocaleString("en-IN")+' of '+S.sampled.toLocaleString("en-IN")+' payments. All fraud is kept.</div>':"");
    return;
  }
  var n=ND[S.sel],o=nstat(S.sel),r=role(n,o);
  var rows=Object.keys(o.by).sort(function(a,b){return o.by[b]-o.by[a];}).map(function(k){return '<dt><span class="sw" style="background:'+typeColor(k)+';margin-right:8px;vertical-align:middle"></span>'+esc(k)+'</dt><dd class="mono">'+o.by[k]+'</dd>';}).join("");
  var links=n.type==="A"?'<dt>Devices used</dt><dd class="mono">'+o.devs.size+'</dd>'+(o.shops.size?'<dt>Shops paid</dt><dd class="mono">'+o.shops.size+'</dd>':'')+(o.upis.size?'<dt>UPI IDs</dt><dd class="mono">'+o.upis.size+'</dd>':''):'<dt>Different accounts</dt><dd class="mono">'+o.accts.size+'</dd>';
  var det=D.hasPred&&o.tn?'<dt>Model, held-out</dt><dd class="mono">'+o.caught+' caught, '+o.missed+' missed, '+o.fa+' false</dd>':"";
  el.innerHTML='<p class="eyebrow" style="margin-bottom:2px">'+kind(n)+'</p><h3 class="mono" style="font-family:var(--font-mono);font-size:20px">'+n.id+'</h3><span class="badge '+r[2]+'" style="margin:4px 0 10px;display:inline-block">'+r[0]+'</span>'+
    '<dl class="kv"><dt>Payments</dt><dd class="mono">'+o.n+'</dd><dt>Fraud payments</dt><dd class="mono">'+o.fr+'</dd>'+(n.type==="A"?'<dt>Fraud sent / received</dt><dd class="mono">'+o.sent+' / '+o.recv+'</dd>':'')+'<dt>Fraud value</dt><dd class="mono">'+inr(o.val)+'</dd>'+links+det+rows+'</dl>'+
    '<div class="callout">'+r[1]+'</div><button class="btn" id="clr" type="button" style="margin-top:12px">Clear selection</button>';
  $("clr").onclick=function(){S.sel=null;drawNet();};
}
function placeTip(e,h){
  var tip=$("tip"),st=$("stage"),r=st.getBoundingClientRect();
  tip.innerHTML=h;tip.style.display="block";
  var w=tip.offsetWidth,ht=tip.offsetHeight,x=e.clientX-r.left+18,y=e.clientY-r.top+18;
  if(x+w>st.clientWidth-8)x=e.clientX-r.left-w-18;
  if(y+ht>st.clientHeight-8)y=e.clientY-r.top-ht-18;
  tip.style.left=Math.max(8,x)+"px";tip.style.top=Math.max(8,y)+"px";
}
function hideTip(){$("tip").style.display="none";}
var SHAPE={
  A:'<svg width="30" height="30" viewBox="0 0 30 30"><circle cx="15" cy="15" r="11" fill="var(--node)"/></svg>',
  D:'<svg width="30" height="30" viewBox="0 0 30 30"><rect x="4" y="4" width="22" height="22" rx="3" fill="var(--node-d)"/></svg>',
  M:'<svg width="30" height="30" viewBox="0 0 30 30"><path d="M15 2l13 13-13 13L2 15z" fill="var(--node-m)"/></svg>',
  U:'<svg width="30" height="30" viewBox="0 0 30 30"><path d="M15 3l13 23H2z" fill="var(--node-u)"/></svg>'
};
function nodeTip(n){
  var o=nstat(n.id),r=role(n,o),rate=o.n?o.fr/o.n:0,rc=r[2]==="bad"?"--tp":r[2]==="warn"?"--fn":"--tn";
  var types=Object.keys(o.by).sort(function(a,b){return o.by[b]-o.by[a];}).map(function(k){return '<div class="tt-row"><i class="sw" style="background:'+typeColor(k)+'"></i><span>'+esc(k)+'</span><b>'+o.by[k]+'</b></div>';}).join("");
  var ctx=n.type==="A"?"Uses "+o.devs.size+" device"+(o.devs.size===1?"":"s")+(o.upis.size?", "+o.upis.size+" UPI ID"+(o.upis.size===1?"":"s"):""):"Used by "+o.accts.size+" account"+(o.accts.size===1?"":"s");
  var det=D.hasPred&&o.tn?'<div class="tt-sub">Model on held-out payments here: caught '+o.caught+', missed '+o.missed+', false alarms '+o.fa+'</div>':"";
  return '<div class="tt-top">'+SHAPE[n.type]+'<div><b>'+kind(n)+' '+n.id+'</b><span class="badge '+r[2]+'">'+r[0]+'</span></div></div>'+
    '<div class="tt-risk"><div class="tt-l"><span>Fraud share of its payments</span><b>'+pct(rate,0)+'</b></div><div class="meter"><i style="width:'+Math.max(2,rate*100)+'%;background:var('+rc+')"></i></div><div class="tt-l" style="margin:4px 0 0"><span>Overall: '+pct(D.base,0)+'</span><span>'+ctx+'</span></div></div>'+
    '<div class="tt-grid"><div><span>Payments</span><b>'+o.n+'</b></div><div><span>Fraud</span><b>'+o.fr+'</b></div><div><span>Fraud value</span><b>'+inr(o.val)+'</b></div></div>'+
    (o.fr?(n.type==="A"?'<div class="tt-sub">Fraud received '+o.recv+', sent '+o.sent+'</div>':'')+types:'')+det+
    '<p class="tt-read">'+r[1]+'</p><div class="tt-foot">Click to focus on this one. Drag to move it.</div>';
}
function edgeTip(t){
  var a=ND[t.s],b=ND[t.r],c=t.fraud?typeColor(t.fraud):css("--legit"),det="";
  if(D.hasPred&&t.pr!=null){
    var f=flag(t);
    det='<div class="tt-sub">Model fraud score <b>'+t.pr.toFixed(2)+'</b> ('+(f?"flagged":"not flagged")+' at '+S.th.toFixed(2)+')</div><div class="tt-sub">'+(t.test?"Held-out test payment: "+OUT[outcome(t)][0].toLowerCase():"Used to train or tune the model, so not a fair test")+'</div>';
  }
  return '<div class="tt-amt">'+inr(t.amt)+'</div><span class="badge" style="background:color-mix(in srgb,'+c+' 18%,var(--surface))"><i class="sw" style="background:'+c+';width:10px;height:4px;margin-right:6px;vertical-align:middle"></i>'+(t.fraud?"Fraud: "+esc(t.fraud):"Normal payment")+'</span>'+
    '<div class="tt-sub" style="margin-top:10px">From '+kind(a).toLowerCase()+' <b>'+a.id+'</b> to '+kind(b).toLowerCase()+' <b>'+b.id+'</b></div>'+
    '<div class="tt-sub">'+esc(t.ch||"Unknown")+(t.ts?" on "+fmtDate(t.ts)+" at "+hh(t.hr)+":00":" at "+hh(t.hr)+":00")+'</div>'+det+(t.fraud&&TY[t.fraud]?'<p class="tt-read">'+TY[t.fraud][1]+'</p>':'');
}

/* ---------- Network: graph ---------- */
function edgeColor(t){return S.mode==="label"?(t.fraud?typeColor(t.fraud):css("--legit")):css(OUT[outcome(t)][1]);}
function edgeOpacity(t){if(S.mode==="label")return t.fraud?.9:.2;var o=outcome(t);return o==="TR"?.1:o==="TN"?.22:.95;}
function edgeWidth(t){var w=.5+Math.max(0,Math.log10(Math.max(t.amt,10))-1)*.38;return t.fraud||(S.mode==="outcome"&&outcome(t)==="FP")?w+.6:w;}
function nodePath(d){
  var s=Math.PI*d.r0*d.r0,T=d.type==="M"?d3.symbolDiamond:d.type==="D"?d3.symbolSquare:d.type==="U"?d3.symbolTriangle:d3.symbolCircle;
  return d3.symbol().type(T).size(d.type==="M"?s*1.3:d.type==="U"?s*1.2:s)();
}
function nodeFill(d){return css(d.type==="M"?"--node-m":d.type==="D"?"--node-d":d.type==="U"?"--node-u":"--node");}
function initNet(){
  svg=d3.select("#net");
  gRoot=svg.append("g");
  gRoot.append("g").attr("class","links");gRoot.append("g").attr("class","halos");gRoot.append("g").attr("class","nodes");
  zoom=d3.zoom().scaleExtent([.15,6]).on("zoom",function(e){gRoot.attr("transform",e.transform);});
  svg.call(zoom).on("dblclick.zoom",null);
  svg.on("click",function(e){if(e.target===svg.node()){S.sel=null;drawNet();}});
  sim=d3.forceSimulation().force("link",d3.forceLink().id(function(d){return d.id;}).distance(function(l){return l.t?58:42;}).strength(function(l){return l.t?.28:.5;}))
    .force("charge",d3.forceManyBody().strength(-150).distanceMax(420)).force("collide",d3.forceCollide().radius(function(d){return d.r0+7;}))
    .on("tick",ticked);
}
function ticked(){
  if(linkSel)linkSel.attr("x1",function(d){return d.source.x;}).attr("y1",function(d){return d.source.y;}).attr("x2",function(d){return d.target.x;}).attr("y2",function(d){return d.target.y;});
  if(nodeSel)nodeSel.attr("transform",function(d){return "translate("+d.x+","+d.y+")";});
  if(haloSel)haloSel.attr("cx",function(d){return d.x;}).attr("cy",function(d){return d.y;});
}
function styleGraph(hid){
  var focus=hid||S.sel,nb=focus&&adj.has(focus)?adj.get(focus):null;
  if(linkSel)linkSel.attr("stroke-opacity",function(l){var o=l.t?edgeOpacity(l.t):.4;if(nb){var inc=eid(l.source)===focus||eid(l.target)===focus;o=inc?Math.max(o,.9):o*.08;}return o;});
  var dim=function(d){return nb&&d.id!==focus&&!nb.has(d.id)?.15:1;};
  if(nodeSel)nodeSel.attr("opacity",dim);
  if(haloSel)haloSel.attr("opacity",dim);
}
function fitGraph(){
  if(!svg||!sim)return;
  var el=$("net"),W=el.clientWidth,H=el.clientHeight,ns=sim.nodes();
  if(!ns.length||!W)return;
  var x0=d3.min(ns,function(d){return d.x-d.r0;}),x1=d3.max(ns,function(d){return d.x+d.r0;}),y0=d3.min(ns,function(d){return d.y-d.r0;}),y1=d3.max(ns,function(d){return d.y+d.r0;});
  var top=112,bw=Math.max(1,x1-x0),bh=Math.max(1,y1-y0),k=Math.min(1.6,.94*Math.min(W/bw,(H-top-16)/bh));
  var tx=W/2-k*(x0+x1)/2,ty=top+(H-top-16)/2-k*(y0+y1)/2;
  svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity.translate(tx,ty).scale(k));
}
function drawGraph(fit){
  legend();
  var el=$("net"),W=el.clientWidth||800,H=el.clientHeight||600;
  if(el.clientWidth){lastW=el.clientWidth;lastH=el.clientHeight;}
  var vt=visible(),links=vt.map(function(t){return {source:t.s,target:t.r,t:t,k:t.id};}),seen=new Set();
  if(S.links)vt.forEach(function(t){[t.dev,t.m,t.u].forEach(function(x){if(!x)return;var k=t.s+"|"+x;if(seen.has(k))return;seen.add(k);links.push({source:t.s,target:x,t:null,k:"L"+k});});});
  var ids=new Set(),deg={};adj=new Map();
  links.forEach(function(l){
    ids.add(l.source);ids.add(l.target);
    deg[l.source]=(deg[l.source]||0)+1;deg[l.target]=(deg[l.target]||0)+1;
    if(!adj.has(l.source))adj.set(l.source,new Set());if(!adj.has(l.target))adj.set(l.target,new Set());
    adj.get(l.source).add(l.target);adj.get(l.target).add(l.source);
  });
  var nodes=Array.from(ids).map(function(id){
    var o=cache.get(id);if(!o){o=Object.assign({x:W/2+(Math.random()-.5)*W*.6,y:H/2+(Math.random()-.5)*H*.6},ND[id]);cache.set(id,o);}
    o.deg=deg[id]||0;o.r0=4+Math.sqrt(o.deg)*1.5;return o;
  });
  linkSel=gRoot.select(".links").selectAll("line").data(links,function(l){return l.k;}).join("line").attr("class","link").attr("stroke-linecap","round")
    .attr("stroke",function(l){return l.t?edgeColor(l.t):css("--link");})
    .attr("stroke-width",function(l){return l.t?edgeWidth(l.t):.8;})
    .attr("stroke-dasharray",function(l){return l.t?null:"3 3";})
    .style("pointer-events",function(l){return l.t?null:"none";})
    .on("mousemove",function(e,l){if(l.t)placeTip(e,edgeTip(l.t));}).on("mouseleave",hideTip);
  var missed=new Set();if(S.mode==="outcome")vt.forEach(function(t){if(outcome(t)==="FN"){missed.add(t.s);missed.add(t.r);}});
  var hl=nodes.filter(function(d){return S.hot.has(d.id);});
  haloSel=gRoot.select(".halos").selectAll("circle").data(hl,function(d){return d.id;}).join("circle")
    .attr("r",function(d){return d.r0+10;}).attr("fill","var(--tp)").attr("fill-opacity",.12).attr("stroke","var(--tp)").attr("stroke-opacity",.55).attr("stroke-width",1.2).style("pointer-events","none");
  nodeSel=gRoot.select(".nodes").selectAll("path").data(nodes,function(d){return d.id;}).join("path")
    .attr("d",nodePath).attr("fill",nodeFill)
    .attr("stroke",function(d){return d.id===S.sel?css("--accent"):missed.has(d.id)?css("--fn"):css("--canvas");})
    .attr("stroke-width",function(d){return d.id===S.sel?3:missed.has(d.id)?2.4:1.4;})
    .style("cursor","pointer")
    .on("click",function(e,d){e.stopPropagation();S.sel=S.sel===d.id?null:d.id;drawNet();})
    .on("mouseenter",function(e,d){styleGraph(d.id);})
    .on("mousemove",function(e,d){placeTip(e,nodeTip(d));})
    .on("mouseleave",function(){hideTip();styleGraph(null);})
    .call(d3.drag().on("start",function(e,d){if(!e.active)sim.alphaTarget(.25).restart();d.fx=d.x;d.fy=d.y;})
      .on("drag",function(e,d){d.fx=e.x;d.fy=e.y;}).on("end",function(e,d){if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));
  sim.force("center",d3.forceCenter(W/2,H/2).strength(.05));
  sim.nodes(nodes);sim.force("link").links(links);sim.alpha(.8).restart();
  styleGraph(null);drawSide(vt);
  if(fit){clearTimeout(fitT);fitT=setTimeout(fitGraph,1400);}
}

/* ---------- Network: other pictures ---------- */
function fraudTx(){return D.txns.filter(function(t){return t.fraud&&S.types.has(t.fraud);});}
function emptyMsg(sv,W,H,msg){sv.append("text").attr("x",W/2).attr("y",H/2).attr("text-anchor","middle").attr("fill","var(--muted)").attr("font-size",14).text(msg);return {legend:""};}
function leg(note,items){return '<span style="flex-basis:100%;color:var(--ink)">'+note+'</span>'+items;}
function typeLeg(){return typesList().filter(function(t){return S.types.has(t);}).map(function(ty){return '<span><i class="sw" style="background:'+typeColor(ty)+'"></i>'+esc(ty)+'</span>';}).join("");}

function drawClusters(sv,W,H){
  var fr=fraudTx(),by={};
  fr.forEach(function(t){[t.s,t.r].forEach(function(id){var o=by[id]||(by[id]={id:id,by:{},n:0});o.by[t.fraud]=(o.by[t.fraud]||0)+1;o.n++;});});
  var groups={};
  Object.keys(by).forEach(function(id){var o=by[id],best=Object.keys(o.by).sort(function(a,b){return o.by[b]-o.by[a]||(a<b?-1:1);})[0];(groups[best]=groups[best]||[]).push(o);});
  var kids=Object.keys(groups).sort().map(function(k){return {name:k,children:groups[k].map(function(o){return {id:o.id,v:o.n};})};});
  if(!kids.length)return emptyMsg(sv,W,H,"No fraud to show. Turn a kind of fraud back on.");
  var root=d3.hierarchy({name:"root",children:kids}).sum(function(d){return d.v||0;}).sort(function(a,b){return b.value-a.value;});
  d3.pack().size([W-24,H-96]).padding(function(d){return d.depth===0?46:3;})(root);
  var g=sv.append("g").attr("transform","translate(12,64)");
  root.children.forEach(function(grp){
    var c=typeColor(grp.data.name);
    g.append("circle").attr("cx",grp.x).attr("cy",grp.y).attr("r",grp.r).attr("fill",c).attr("fill-opacity",.07).attr("stroke",c).attr("stroke-opacity",.5).attr("stroke-dasharray","4 5");
    g.append("text").attr("x",grp.x).attr("y",grp.y+grp.r+18).attr("text-anchor","middle").attr("font-size",13).attr("font-weight",600).attr("fill","var(--ink)").text(grp.data.name+" · "+grp.children.length+" accounts");
  });
  g.selectAll("circle.leaf").data(root.leaves()).join("circle").attr("class","leaf")
    .attr("cx",function(d){return d.x;}).attr("cy",function(d){return d.y;}).attr("r",function(d){return d.r;})
    .attr("fill",function(d){return typeColor(d.parent.data.name);}).attr("fill-opacity",.85)
    .attr("stroke",function(d){return d.data.id===S.sel?css("--accent"):css("--canvas");}).attr("stroke-width",function(d){return d.data.id===S.sel?3:1.2;}).style("cursor","pointer")
    .on("mousemove",function(e,d){placeTip(e,nodeTip(ND[d.data.id]));}).on("mouseleave",hideTip)
    .on("click",function(e,d){S.sel=S.sel===d.data.id?null:d.data.id;drawNet();});
  g.selectAll("text.lb").data(root.leaves().filter(function(d){return d.r>=17;})).join("text").attr("class","lb")
    .attr("x",function(d){return d.x;}).attr("y",function(d){return d.y;}).attr("text-anchor","middle").attr("dominant-baseline","central").attr("font-size",11).attr("fill","var(--surface)").style("pointer-events","none")
    .text(function(d){return d.data.id;});
  return {legend:leg("Each bubble is an account touched by fraud. Bigger means more fraud payments. Bubbles sit in the group of the fraud kind they see most.",typeLeg())};
}
function drawFlow(sv,W,H){
  var fr=fraudTx(),rv={},tv={},rt={},link={},cntL={};
  if(!fr.length)return emptyMsg(sv,W,H,"No fraud to show. Turn a kind of fraud back on.");
  fr.forEach(function(t){rv[t.r]=(rv[t.r]||0)+t.amt;});
  var tops=Object.keys(rv).sort(function(a,b){return rv[b]-rv[a];}).slice(0,8),topSet=new Set(tops);
  fr.forEach(function(t){var rk=topSet.has(t.r)?t.r:"__other",k=t.fraud+"|"+rk;link[k]=(link[k]||0)+t.amt;cntL[k]=(cntL[k]||0)+1;tv[t.fraud]=(tv[t.fraud]||0)+t.amt;rt[rk]=(rt[rk]||0)+t.amt;});
  var types=Object.keys(tv).sort(function(a,b){return tv[b]-tv[a];}),right=tops.concat(["__other"]).filter(function(r){return rt[r];});
  var total=sum(types.map(function(t){return tv[t];})),gapL=18,gapR=12,top=72,avail=H-top-24;
  var k=Math.min((avail-gapL*(types.length-1))/total,(avail-gapR*(right.length-1))/total);
  var narrow=W<700,xl=narrow?96:190,wL=16,xr=W-(narrow?130:250),ly={},ry={};
  var offL=(avail-(total*k+gapL*(types.length-1)))/2+top,offR=(avail-(total*k+gapR*(right.length-1)))/2+top;
  types.forEach(function(t){ly[t]=offL;offL+=tv[t]*k+gapL;});
  right.forEach(function(r){ry[r]=offR;offR+=rt[r]*k+gapR;});
  var loff={},roff={};types.forEach(function(t){loff[t]=ly[t];});right.forEach(function(r){roff[r]=ry[r];});
  var lab=function(r){return r==="__other"?"All other receivers":kind(ND[r])+" "+r;};
  var gl=sv.append("g");
  types.forEach(function(ty){right.forEach(function(rk){
    var v=link[ty+"|"+rk];if(!v)return;
    var h=Math.max(1.5,v*k),y0=loff[ty],y1=roff[rk],x0=xl+wL,xm=(x0+xr)/2;loff[ty]+=h;roff[rk]+=h;
    gl.append("path").attr("d","M"+x0+" "+y0+"C"+xm+" "+y0+","+xm+" "+y1+","+xr+" "+y1+"L"+xr+" "+(y1+h)+"C"+xm+" "+(y1+h)+","+xm+" "+(y0+h)+","+x0+" "+(y0+h)+"Z")
      .attr("fill",typeColor(ty)).attr("fill-opacity",.4)
      .on("mousemove",function(e){d3.select(this).attr("fill-opacity",.8);placeTip(e,'<div class="tt-amt">'+inr(v)+'</div><p class="tt-read" style="margin:6px 0 0">'+cntL[ty+"|"+rk]+' <b>'+esc(ty)+'</b> payments ended at <b>'+lab(rk)+'</b>.</p>');})
      .on("mouseleave",function(){d3.select(this).attr("fill-opacity",.4);hideTip();});
  });});
  types.forEach(function(ty){
    var h=Math.max(4,tv[ty]*k),y=ly[ty];
    sv.append("rect").attr("x",xl).attr("y",y).attr("width",wL).attr("height",h).attr("rx",4).attr("fill",typeColor(ty));
    sv.append("text").attr("x",xl-12).attr("y",y+h/2-8).attr("text-anchor","end").attr("font-size",13).attr("font-weight",600).attr("fill","var(--ink)").text(ty);
    sv.append("text").attr("x",xl-12).attr("y",y+h/2+10).attr("text-anchor","end").attr("font-size",12).attr("fill","var(--muted)").text(inr(tv[ty]));
  });
  right.forEach(function(rk){
    var h=Math.max(4,rt[rk]*k),y=ry[rk],isO=rk==="__other";
    sv.append("rect").attr("x",xr).attr("y",y).attr("width",wL).attr("height",h).attr("rx",4).attr("fill",isO?css("--legit"):nodeFill(ND[rk]))
      .attr("stroke",rk===S.sel?css("--accent"):"none").attr("stroke-width",2).style("cursor",isO?"default":"pointer")
      .on("mousemove",function(e){if(!isO)placeTip(e,nodeTip(ND[rk]));}).on("mouseleave",hideTip)
      .on("click",function(){if(!isO){S.sel=S.sel===rk?null:rk;drawNet();}});
    sv.append("text").attr("x",xr+wL+10).attr("y",y+h/2-(narrow||h<34?0:8)).attr("dominant-baseline","central").attr("font-size",13).attr("font-weight",600).attr("fill","var(--ink)").text(lab(rk)+(h<34&&!narrow?"  \u00B7  "+inr(rt[rk]):""));
    if(!narrow&&h>=34)sv.append("text").attr("x",xr+wL+10).attr("y",y+h/2+10).attr("dominant-baseline","central").attr("font-size",12).attr("fill","var(--muted)").text(inr(rt[rk]));
  });
  sv.append("text").attr("x",xl).attr("y",top-18).attr("font-size",12).attr("fill","var(--muted)").text("KIND OF FRAUD");
  sv.append("text").attr("x",xr).attr("y",top-18).attr("font-size",12).attr("fill","var(--muted)").text("WHERE THE MONEY ENDED UP");
  return {legend:leg("Follow the money. The left column is each kind of fraud and the right column is the accounts that received the most stolen money. Thicker bands mean more money.","")};
}
function drawTiming(sv,W,H){
  var types=typesList().filter(function(t){return S.types.has(t);});
  if(!types.length)return emptyMsg(sv,W,H,"No fraud to show. Turn a kind of fraud back on.");
  var rows=types.concat(["__n"]),cnt={};
  rows.forEach(function(r){cnt[r]=new Array(24).fill(0);});
  D.txns.forEach(function(t){var r=t.fraud?(S.types.has(t.fraud)?t.fraud:null):"__n";if(r)cnt[r][Math.max(0,Math.min(23,t.hr|0))]++;});
  var labW=Math.min(180,W*.24),top=80,cw=Math.max(12,(W-labW-28)/24),rh=Math.max(34,Math.min(62,(H-top-140)/rows.length));
  for(var h=0;h<24;h+=3)sv.append("text").attr("x",labW+h*cw+cw/2).attr("y",top-14).attr("text-anchor","middle").attr("font-size",11).attr("fill","var(--muted)").attr("class","mono").text(hh(h));
  rows.forEach(function(r,ri){
    var y=top+ri*rh,mx=Math.max.apply(null,cnt[r])||1,tot=sum(cnt[r]),col=r==="__n"?css("--legit"):typeColor(r);
    sv.append("text").attr("x",labW-12).attr("y",y+rh/2).attr("text-anchor","end").attr("dominant-baseline","central").attr("font-size",13).attr("font-weight",600).attr("fill","var(--ink)").text(r==="__n"?"Normal payments":r);
    cnt[r].forEach(function(c,hr){
      sv.append("rect").attr("x",labW+hr*cw+1.5).attr("y",y+3).attr("width",cw-3).attr("height",rh-6).attr("rx",6)
        .style("fill","color-mix(in srgb,"+col+" "+Math.round(c?16+78*c/mx:0)+"%,var(--surface2))")
        .on("mousemove",function(e){placeTip(e,'<div class="tt-amt">'+hh(hr)+':00</div><p class="tt-read" style="margin:6px 0 0"><b>'+c+'</b> '+(r==="__n"?"normal":esc(r))+' payments in this hour ('+pct(c/(tot||1))+' of all its payments).</p>');}).on("mouseleave",hideTip);
    });
  });
  var fa=new Array(24).fill(0);types.forEach(function(r){cnt[r].forEach(function(v,i){fa[i]+=v;});});
  var ft=sum(fa),pk=fa.indexOf(Math.max.apply(null,fa)),nt=sum(cnt.__n);
  var nightF=sum(fa.slice(0,6))/(ft||1),nightN=sum(cnt.__n.slice(0,6))/(nt||1),yb=top+rows.length*rh+34;
  sv.append("text").attr("x",labW).attr("y",yb).attr("font-size",15).attr("font-weight",600).attr("fill","var(--ink)").text("Fraud is busiest at "+hh(pk)+":00, with "+fa[pk]+" payments.");
  sv.append("text").attr("x",labW).attr("y",yb+24).attr("font-size",13).attr("fill","var(--muted)").text(pct(nightF)+" of fraud happens between midnight and 6 am, against "+pct(nightN)+" of normal payments.");
  if(Math.abs(nightF-nightN)<.03)sv.append("text").attr("x",labW).attr("y",yb+46).attr("font-size",13).attr("font-weight",600).attr("fill","var(--ink)").text("Hour of day barely separates fraud from normal payments in this data, so it is not a useful clue.");
  return {legend:leg("When does fraud happen? Each square is one hour of the day. Darker means more payments in that hour. Each row is scaled to its own busiest hour.","")};
}
function drawAlt(){
  var el=$("alt"),W=el.clientWidth||800,H=el.clientHeight||600,sv=d3.select(el);
  sv.selectAll("*").remove();sv.attr("viewBox","0 0 "+W+" "+H);
  var info=({cl:drawClusters,fl:drawFlow,tm:drawTiming})[S.view](sv,W,H);
  $("legend").innerHTML=info.legend;
  drawSide(visible());
}
function drawNet(fit){
  if(!D||$("v-net").hidden)return;
  if(!svg)initNet();
  if(S.view==="net")drawGraph(fit);else drawAlt();
}
function setView(v){
  S.view=v;
  ["net","cl","fl","tm"].forEach(function(k){$("vw-"+k).setAttribute("aria-pressed",k===v?"true":"false");});
  var g=v==="net";
  [["net",!g],["alt",g]].forEach(function(p){if(p[1])$(p[0]).setAttribute("hidden","");else $(p[0]).removeAttribute("hidden");});$("tb-net").hidden=!g;$("zoombox").hidden=!g;
  hideTip();drawNet(true);
}

/* ---------- import ---------- */
function parseCSV(text){
  var first=text.split(/\r?\n/,1)[0],d=first.indexOf("\t")>-1?"\t":(first.indexOf(";")>-1&&first.indexOf(",")<0?";":",");
  var rows=[],row=[],f="",q=false;
  for(var i=0;i<text.length;i++){var c=text.charAt(i);
    if(q){if(c==='"'){if(text.charAt(i+1)==='"'){f+='"';i++;}else q=false;}else f+=c;}
    else if(c==='"')q=true;
    else if(c===d){row.push(f);f="";}
    else if(c==="\n"||c==="\r"){if(c==="\r"&&text.charAt(i+1)==="\n")i++;row.push(f);rows.push(row);row=[];f="";}
    else f+=c;}
  if(f.length||row.length){row.push(f);rows.push(row);}
  return rows.filter(function(r){return r.length>1||r[0]!=="";});
}
function csvToData(text){
  var rows=parseCSV(text);if(rows.length<2)throw new Error("The file needs a header row and at least one data row.");
  var norm=function(h){return String(h).toLowerCase().replace(/[^a-z0-9]/g,"");},idx={};
  rows[0].forEach(function(h,i){idx[norm(h)]=i;});
  var col=function(names){for(var k=0;k<names.length;k++)if(idx[names[k]]!=null)return idx[names[k]];return -1;};
  var c={id:col(["transactionid","id","txnid"]),amt:col(["amount","amt"]),s:col(["senderaccount","senderaccountid","sender"]),r:col(["receiveraccount","receiveraccountid","receiver"]),
    dev:col(["deviceid","device"]),m:col(["merchantid","merchant"]),u:col(["upiid","upi"]),ch:col(["channel"]),hr:col(["transactionhour","hour","hr"]),ts:col(["transactiontimestamp","timestamp"]),
    fr:col(["fraudtype","fraud"]),lb:col(["label","isfraud"]),p:col(["predictedprob","predictedprobability","probability","fraudscore","score"]),pl:col(["predictedlabel","predicted","prediction"]),
    sp:col(["split","dataset"]),te:col(["istest","test","heldout"])};
  var miss=[];if(c.id<0)miss.push("Transaction_ID");if(c.amt<0)miss.push("Amount");if(c.s<0)miss.push("Sender_Account");if(c.r<0&&c.m<0)miss.push("Receiver_Account or Merchant_ID");
  if(miss.length)throw new Error("Missing column: "+miss.join(", ")+".");
  var nul=function(v){if(v==null)return null;v=String(v).trim();return v===""||/^(null|none|nan)$/i.test(v)?null:v;};
  var txns=[],skipped=0,hasSplit=c.sp>=0||c.te>=0,hasPred=c.p>=0||c.pl>=0;
  for(var i=1;i<rows.length;i++){
    var r=rows[i],g=function(k){return k<0?null:nul(r[k]);};
    var amt=parseFloat(String(g(c.amt)).replace(/[,\s₹]/g,"")),s=g(c.s),ra=g(c.r),me=g(c.m),id=g(c.id);
    var recv=ra?"A"+ra:me?"M"+me:null;
    if(!(amt>=0)||!s||!recv||!id){skipped++;continue;}
    var fr=g(c.fr),lb=g(c.lb);
    if(fr&&/^no ?fraud$/i.test(fr))fr=null;
    if(!fr&&lb&&/^(1|true|yes)$/i.test(lb))fr="Fraud";
    var pr=null;
    if(c.p>=0){var pv=parseFloat(g(c.p));pr=isNaN(pv)?null:Math.min(1,Math.max(0,pv));}
    else if(c.pl>=0){var pl=g(c.pl);pr=pl==null?null:(/^(1|true|yes|fraud)$/i.test(pl)?1:0);}
    var sp=0;
    if(c.sp>=0){var sv=g(c.sp)||"";sp=/^te/i.test(sv)?2:/^va/i.test(sv)?1:0;}
    else if(c.te>=0)sp=/^(1|true|t|yes|test)$/i.test(g(c.te)||"")?2:0;
    else if(hasPred)sp=2;
    var ts=null,hr=g(c.hr),h=hr!=null?parseInt(hr,10):0;
    if(c.ts>=0){var mt=String(r[c.ts]).match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2})/);if(mt){ts=mt[1]+"T"+mt[2];if(hr==null)h=parseInt(mt[2],10);}}
    var dv=g(c.dev),mc=g(c.m),up=g(c.u);
    txns.push({id:id,s:"A"+s,r:recv,dev:dv?"D"+dv:null,m:mc?"M"+mc:null,u:up?"U"+up:null,amt:amt,ch:g(c.ch)||"Unknown",ts:ts,hr:isNaN(h)?0:h,fraud:fr,pr:pr,sp:sp,test:sp===2});
  }
  if(!txns.length)throw new Error("No usable rows. Check Sender_Account, Amount and the receiver columns.");
  return {txns:txns,hasPred:hasPred,fixed:c.p<0&&c.pl>=0,skipped:skipped,noSplit:hasPred&&!hasSplit,th0:c.p>=0?.5:.5};
}
function loadFile(f){
  var err=$("fileerr");if(!f)return;
  var rd=new FileReader();
  rd.onload=function(){
    try{
      var txt=String(rd.result).replace(/^﻿/,""),t0=txt.trim().charAt(0),j,extra="";
      if(t0==="{"||t0==="["){
        j=JSON.parse(txt);if(Array.isArray(j))j={txns:j};
        if(!Array.isArray(j.txns)||!j.txns.length)throw new Error("Expected a non-empty txns list.");
        j.txns.forEach(function(t){if(t.id==null||t.s==null||t.r==null||typeof t.amt!=="number")throw new Error("Each payment needs id, s, r and a numeric amt.");
          t.id=String(t.id);t.s=String(t.s);t.r=String(t.r);t.ch=t.ch||"Unknown";t.hr=t.hr||0;t.fraud=t.fraud||null;t.pr=typeof t.pr==="number"?t.pr:null;t.test=!!t.test;t.ts=t.ts||null;t.dev=t.dev||null;t.m=t.m||null;t.u=t.u||null;});
        j.hasPred=j.txns.some(function(t){return t.pr!=null;});j.th0=.5;
      }else{
        j=csvToData(txt);
        if(j.skipped)extra+=" "+j.skipped+" rows were skipped for missing values.";
        if(j.noSplit)extra+=" No Split or Is_Test column, so every row counts as held out.";
      }
      j.project=false;setData(j);
      err.textContent="Loaded "+j.txns.length.toLocaleString("en-IN")+" payments from "+f.name+"."+extra+(D.hasPred?"":" No model scores found, so accuracy is not shown.");
    }catch(x){err.textContent="Could not load file: "+x.message;}
  };
  rd.readAsText(f);
}

/* ---------- wiring ---------- */
function show(v){
  ["home","net"].forEach(function(k){$("v-"+k).hidden=k!==v;$("tab-"+k).setAttribute("aria-selected",k===v?"true":"false");});
  if(v==="net"){if(!svg)initNet();drawNet(true);}
}
document.querySelectorAll(".tabs button").forEach(function(b){b.addEventListener("click",function(){show(b.dataset.v);});});
$("v-all").addEventListener("click",function(){S.types.add(NONE);syncControls();drawNet(true);});
$("v-fraud").addEventListener("click",function(){S.types.delete(NONE);S.mode="label";syncControls();drawNet(true);});
$("f-miss").addEventListener("change",function(e){S.mode=e.target.checked?"outcome":"label";if(e.target.checked)S.types.add(NONE);syncControls();drawNet(true);});
$("f-links").addEventListener("change",function(e){S.links=e.target.checked;drawNet(true);});
$("f-reset").addEventListener("click",function(){
  S.sel=null;S.mode="label";S.links=false;S.types=new Set(typesList());syncControls();
  if(svg)svg.call(zoom.transform,d3.zoomIdentity);drawNet(true);
});
$("zfit").addEventListener("click",fitGraph);
["net","cl","fl","tm"].forEach(function(k){$("vw-"+k).addEventListener("click",function(){setView(k);});});
$("zin").addEventListener("click",function(){if(svg)svg.transition().duration(220).call(zoom.scaleBy,1.4);});
$("zout").addEventListener("click",function(){if(svg)svg.transition().duration(220).call(zoom.scaleBy,1/1.4);});
var thRaf=0;
$("th").addEventListener("input",function(e){S.th=parseFloat(e.target.value);cancelAnimationFrame(thRaf);thRaf=requestAnimationFrame(drawHome);});
$("th-reset").addEventListener("click",function(){S.th=D.th0;drawHome();});
$("file").addEventListener("change",function(e){loadFile(e.target.files[0]);});
var drop=$("drop");
["dragenter","dragover"].forEach(function(ev){drop.addEventListener(ev,function(e){e.preventDefault();drop.classList.add("over");});});
["dragleave","drop"].forEach(function(ev){drop.addEventListener(ev,function(e){e.preventDefault();drop.classList.remove("over");});});
drop.addEventListener("drop",function(e){var f=e.dataTransfer&&e.dataTransfer.files[0];if(f)loadFile(f);});
$("demo-btn").addEventListener("click",function(){setData(fromProject());$("fileerr").textContent="Back on the project results.";});
var snip="sql";
function setSnip(k){snip=k;$("snip-sql").hidden=k!=="sql";$("snip-py").hidden=k!=="py";$("sn-sql").setAttribute("aria-pressed",k==="sql"?"true":"false");$("sn-py").setAttribute("aria-pressed",k==="py"?"true":"false");}
$("sn-sql").addEventListener("click",function(){setSnip("sql");});
$("sn-py").addEventListener("click",function(){setSnip("py");});
$("copy").addEventListener("click",function(){
  var b=this,node=$("snip-"+snip);
  function done(ok){b.textContent=ok?"Copied":"Press Ctrl+C";setTimeout(function(){b.textContent="Copy";},1800);}
  function fallback(){var rg=document.createRange();rg.selectNodeContents(node);var sel=window.getSelection();sel.removeAllRanges();sel.addRange(rg);done(false);}
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(node.textContent).then(function(){done(true);},fallback);}else fallback();
});
function reflow(){
  if(!svg||$("v-net").hidden)return;
  if(S.view!=="net"){drawAlt();return;}
  var el=$("net"),W=el.clientWidth,H=el.clientHeight;
  if(!W||!H)return;
  if(lastW&&(Math.abs(W-lastW)>1||Math.abs(H-lastH)>1)){
    var dx=(W-lastW)/2,dy=(H-lastH)/2;
    cache.forEach(function(o){o.x+=dx;o.y+=dy;if(o.fx!=null){o.fx+=dx;o.fy+=dy;}});
    sim.force("center",d3.forceCenter(W/2,H/2).strength(.05));
    sim.alpha(.25).restart();
  }
  lastW=W;lastH=H;
}
function queueReflow(){cancelAnimationFrame(raf);raf=requestAnimationFrame(reflow);}
window.addEventListener("resize",queueReflow);
document.addEventListener("fullscreenchange",function(){setTimeout(queueReflow,120);});
if(window.ResizeObserver)new ResizeObserver(queueReflow).observe($("stage"));
/* ---------- payment look-up: asks the API when live, otherwise answers from the snapshot ---------- */
function localPayment(id){
  var t=D.txns.filter(function(x){return String(x.id)===String(id);})[0];if(!t)return null;
  return {id:t.id,sender:t.s,receiver:t.r,amount:t.amt,channel:t.ch,timestamp:t.ts,fraud:!!t.fraud,fraudType:t.fraud,score:t.pr,split:["train","validation","test"][t.sp]||(t.test?"test":"train"),topFeatures:null};
}
function renderPayment(p){
  var out=$("lk-out");
  if(!p){out.innerHTML='<p class="note" style="margin-top:14px">No payment with that ID. Try a number between 1 and '+D.txns.length+'.</p>';return;}
  var sc=p.score==null?null:p.score,fl=sc!=null&&sc>=S.th,tst=p.split==="test",verdict,cls;
  if(sc==null){verdict="No model score";cls="warn";}
  else if(p.fraud&&fl){verdict="Caught: real fraud, flagged";cls="ok";}
  else if(p.fraud){verdict="Missed: real fraud, not flagged";cls="bad";}
  else if(fl){verdict="False alarm: normal payment, flagged";cls="warn";}
  else{verdict="Correct: normal payment, not flagged";cls="ok";}
  var col=cls==="bad"?"--tp":cls==="warn"?"--fn":"--tn";
  var kv=function(k,v){return '<dt>'+k+'</dt><dd class="mono">'+v+'</dd>';};
  out.innerHTML='<div class="lk-res"><div class="lk-score"><span class="eyebrow">Fraud score</span><div class="v mono">'+(sc==null?"n/a":sc.toFixed(4))+'</div>'+
    (sc==null?'':'<div class="meter"><i style="width:'+Math.max(1,sc*100)+'%;background:var('+col+')"></i><b style="left:'+(S.th*100)+'%"></b></div><span class="note" style="margin:0">Black tick: alert threshold '+S.th.toFixed(2)+'</span>')+
    '<br><span class="badge '+cls+'">'+verdict+'</span>'+
    (tst?'':'<p class="note">This payment was used to train or tune the model, so its score is not a fair test.</p>')+'</div>'+
    '<div><dl class="kv">'+kv("Payment",esc(p.id))+kv("From",esc(p.sender))+kv("To",esc(p.receiver))+kv("Amount",inr(p.amount))+kv("Channel",esc(p.channel||"Unknown"))+
    (p.timestamp?kv("When",esc(String(p.timestamp).replace("T"," ").slice(0,16))):"")+kv("Actually",p.fraud?"Fraud"+(p.fraudType?": "+esc(p.fraudType):""):"Normal")+kv("Data split",esc(p.split))+'</dl>'+
    (p.topFeatures?'<p class="note" style="margin:14px 0 4px">Most unusual inputs for the model (scaled values)</p><ul class="lk-top">'+p.topFeatures.slice(0,5).map(function(f){return '<li><span>'+esc(f.name)+'</span><span class="mono">'+f.value.toFixed(2)+'</span></li>';}).join("")+'</ul>':"")+'</div></div>';
}
function lookup(id){
  var src=$("lk-src");
  if(P.live){
    src.textContent="Asking the API...";
    fetch("api/payments/"+encodeURIComponent(id)).then(function(r){return r.status===404?null:r.ok?r.json():Promise.reject(r.status);})
      .then(function(p){src.textContent="Answered by the API";renderPayment(p);}).catch(function(){src.textContent="The API did not answer";});
  }else{src.textContent="Answered from the snapshot";renderPayment(localPayment(id));}
}
$("lk-form").addEventListener("submit",function(e){e.preventDefault();var v=$("lk-id").value.trim();if(v)lookup(v);});
$("reload").addEventListener("click",function(){
  var b=this;b.disabled=true;b.textContent="Reloading...";
  fetch("api/reload",{method:"POST"}).then(function(r){if(!r.ok)throw new Error();return fetch("api/data");}).then(function(r){return r.json();})
    .then(function(d){P=d;P.live=true;setData(fromProject());b.textContent="Reload from model";b.disabled=false;}).catch(function(){b.textContent="Reload failed";b.disabled=false;});
});

/* ---------- start: live API when served by run_app.py, otherwise the offline snapshot ---------- */
function start(live){
  P.live=!!live;setData(fromProject());
  var ex=D.txns.filter(function(t){return t.fraud&&t.test;})[0];if(ex){$("lk-id").value=ex.id;lookup(ex.id);}
}
function boot(){
  var canFetch=location.protocol.indexOf("http")===0&&!window.GML_NO_API;
  if(!canFetch){P=window.GML_DATA;start(false);return;}
  $("headline").textContent="Loading results from the model...";
  fetch("api/data").then(function(r){if(!r.ok)throw new Error(String(r.status));return r.json();}).then(function(d){P=d;start(true);})
    .catch(function(){P=window.GML_DATA;start(false);});
}
boot();
})();
