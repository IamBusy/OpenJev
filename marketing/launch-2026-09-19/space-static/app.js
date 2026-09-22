"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const worlds = Array.from({length:64},(_,i)=>Array.from({length:6},(_,j)=>(i>>(5-j))&1));
  let posterior=null, session=null, examples=null, encodings=0, requestId=0;
  const predicates={red:[0,1],blue:[0,0],circle:[1,0],square:[1,1]};
  const slots={left:0,center:1,right:2};

  function compile(expression){
    if(!expression.trim() || expression.length>2048)throw Error("Use an event of 1–2048 characters.");
    const tokens=[]; const re=/\s*([a-z_]+|[(),])/gy; let pos=0;
    while(pos<expression.trimEnd().length){re.lastIndex=pos;const m=re.exec(expression);if(!m)throw Error("Unsupported character or event syntax.");tokens.push(m[1]);pos=re.lastIndex;}
    if(tokens.length>128)throw Error("Event is too complex.");
    let i=0;
    function take(t){if(tokens[i]!==t)throw Error(`Expected ${t}.`);i++;}
    function atom(){
      if(tokens[i]==="not"){i++;const f=atom();return w=>!f(w);}
      if(tokens[i]==="("){i++;const f=or();take(")");return f;}
      const name=tokens[i++];if(!Object.hasOwn(predicates,name) && name!=="same_color" && name!=="same_shape")throw Error("Use red, blue, circle, square, same_color or same_shape.");
      take("(");const a=tokens[i++];if(!Object.hasOwn(slots,a))throw Error("Use left, center or right.");
      if(Object.hasOwn(predicates,name)){take(")");const [attr,value]=predicates[name];return w=>w[2*slots[a]+attr]===value;}
      take(",");const b=tokens[i++];if(!Object.hasOwn(slots,b))throw Error("Use left, center or right.");take(")");const attr=name==="same_shape"?1:0;return w=>w[2*slots[a]+attr]===w[2*slots[b]+attr];
    }
    function and(){let f=atom();while(tokens[i]==="and"){i++;const a=f,b=atom();f=w=>a(w)&&b(w);}return f;}
    function or(){let f=and();while(tokens[i]==="or"){i++;const a=f,b=and();f=w=>a(w)||b(w);}return f;}
    const f=or();if(i!==tokens.length)throw Error("Unexpected trailing tokens.");return f;
  }
  function probability(expression){const event=compile(expression);return posterior.reduce((s,p,i)=>s+(event(worlds[i])?p:0),0);}
  function query(){
    if(!posterior)return;
    const out=$("answer");out.replaceChildren();out.classList.remove("error");
    try{const event=$("event").value;out.append(document.createTextNode(`P(${event}) = ${(100*probability(event)).toFixed(2)}%`));const note=document.createElement("small");note.textContent="Answered from the stored distribution. No additional image encoding.";out.append(note);}
    catch(e){out.classList.add("error");out.textContent=e.message;}
  }
  function drawBars(){
    $("bars").replaceChildren();
    for(const [label,event] of [["Left object is red","red(left)"],["Right object is red","red(right)"],["Left and right have the same color","same_color(left, right)"],["Both left and right are red","red(left) and red(right)"]]){
      const p=probability(event);const row=document.createElement("div");row.className="bar-row";
      const l=document.createElement("div");l.className="bar-label";const n=document.createElement("span");n.textContent=label;const v=document.createElement("strong");v.textContent=(100*p).toFixed(2)+"%";l.append(n,v);
      const track=document.createElement("div");track.className="track";track.setAttribute("aria-hidden","true");const fill=document.createElement("div");fill.className="fill";fill.style.width=(100*p)+"%";track.append(fill);row.append(l,track);$("bars").append(row);
    }
  }
  async function encode(index){
    const job=++requestId;posterior=null;$("ask").disabled=true;$("answer").textContent="Encoding this view…";
    for(let i=0;i<2;i++){ $("view"+i).disabled=true;$("view"+i).setAttribute("aria-pressed",String(index===i)); }
    $("status").classList.remove("error");$("status").textContent="Running the model on your device…";
    const img=$("scene");img.src=examples[index].image;img.alt=index===0?"Left and center objects occluded; right is a blue circle":"Left is a red circle, center and right are blue circles";
    await img.decode();const canvas=document.createElement("canvas");canvas.width=192;canvas.height=64;const context=canvas.getContext("2d",{willReadFrequently:true});context.drawImage(img,0,0,192,64);
    const rgba=context.getImageData(0,0,192,64).data;const pixels=new Float32Array(3*64*192);
    for(let i=0;i<64*192;i++)for(let c=0;c<3;c++)pixels[c*64*192+i]=rgba[4*i+c]/255;
    const raw=await session.run({pixels:new ort.Tensor("float32",pixels,[1,3,64,192]),prior:new ort.Tensor("float32",Float32Array.from(examples[index].prior),[1,64])});
    if(job!==requestId)return;const p=Array.from(raw.posterior.data);const total=p.reduce((a,b)=>a+b,0);if(!Number.isFinite(total)||total<=0||p.some(x=>x<0||!Number.isFinite(x)))throw Error("The model returned an invalid distribution.");posterior=p.map(v=>v/total);encodings++;
    drawBars();query();$("status").textContent=`Live browser inference · ${encodings} image encoding${encodings===1?"":"s"} this session · 64 possible scenes`;
    $("ask").disabled=false;$("view0").disabled=false;$("view1").disabled=false;
  }
  function failed(e){$("status").classList.add("error");$("status").textContent="Could not load or run the model. "+e.message;$("retry").hidden=false;$("answer").textContent="No model result is available.";}
  async function init(){
    $("retry").hidden=true;$("status").classList.remove("error");$("status").textContent="Loading the published model…";
    try{if(typeof ort==="undefined")throw Error("ONNX Runtime could not load. Check your connection and reload the page.");ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
      const [data,provenance,modelBytes]=await Promise.all([fetch("demo-data.json").then(r=>{if(!r.ok)throw Error("Example data unavailable.");return r.json();}),fetch("provenance.json").then(r=>r.json()),fetch("model.onnx").then(r=>{if(!r.ok)throw Error("Model download unavailable.");return r.arrayBuffer();})]);
      const digest=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",modelBytes))).map(b=>b.toString(16).padStart(2,"0")).join("");if(digest!==provenance.onnx_sha256)throw Error("Model hash verification failed.");
      examples=data.examples;session=await ort.InferenceSession.create(modelBytes,{executionProviders:["wasm"]});await encode(0);
    }catch(e){failed(e);}
  }
  $("query-form").addEventListener("submit",e=>{e.preventDefault();query();});
  document.querySelectorAll("[data-event]").forEach(b=>b.addEventListener("click",()=>{$("event").value=b.dataset.event;query();}));
  for(let i=0;i<2;i++)$("view"+i).addEventListener("click",()=>encode(i).catch(failed));
  $("retry").addEventListener("click",init);init();
})();
