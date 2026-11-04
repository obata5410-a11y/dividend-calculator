(function(root) {
  'use strict';
  const KEY = 'dividend-history-v1';
  const SCOPES = {simulator: 'このシミュレーターの記録', mf_securities: 'マネーフォワードの証券残高'};
  function jstDay(iso) {
    const d = new Date(iso);
    if (!Number.isFinite(d.getTime())) throw new Error('日時が不正です');
    return new Date(d.getTime() + 9*3600000).toISOString().slice(0,10);
  }
  function validRecord(r) {
    if (!r || !Object.hasOwn(SCOPES,r.scope) || !/^\d{4}-\d{2}-\d{2}$/.test(r.date || '')) return false;
    const d = new Date(r.date+'T00:00:00Z');
    if (!Number.isFinite(d.getTime()) || d.toISOString().slice(0,10)!==r.date || r.date>jstDay(new Date().toISOString())) return false;
    if (!Number.isFinite(r.total_value) || r.total_value<0) return false;
    return ['annual_net','annual_gross','coverage'].every(k => r[k] == null || Number.isFinite(r[k]) && r[k]>=0) && (r.coverage==null || r.coverage<=100);
  }
  function clean(r) {
    return {date:r.date,scope:r.scope,total_value:r.total_value,annual_net:r.annual_net ?? null,
      annual_gross:r.annual_gross ?? null,coverage:r.coverage ?? null,source:String(r.source||'保存記録').slice(0,200),
      calculated_at:r.calculated_at || null,method:r.method || 'unknown'};
  }
  function merge(existing, incoming, replace=false) {
    if (!Array.isArray(incoming) || incoming.length>10000 || !incoming.every(validRecord)) throw new Error('履歴の形式・日付・金額を確認してください');
    const map = new Map(existing.filter(validRecord).map(r=>[r.scope+':'+r.date,clean(r)]));
    for (const r of incoming) {
      const key=r.scope+':'+r.date;
      if (replace || !map.has(key)) map.set(key,clean(r));
    }
    return [...map.values()].sort((a,b)=>a.date.localeCompare(b.date)||a.scope.localeCompare(b.scope));
  }
  function targetDay(day, months) {
    const [y,m,d]=day.split('-').map(Number);
    const first=new Date(Date.UTC(y,m-1+months,1));
    const last=new Date(Date.UTC(first.getUTCFullYear(),first.getUTCMonth()+1,0)).getUTCDate();
    return new Date(Date.UTC(first.getUTCFullYear(),first.getUTCMonth(),Math.min(d,last))).toISOString().slice(0,10);
  }
  function baseline(records,current,months) {
    const target=targetDay(current.date,months), month=target.slice(0,7);
    // Same target month only; do not silently substitute an unrelated period.
    return records.filter(r=>r.scope===current.scope && r.date.slice(0,7)===month && r.date<=target)
      .sort((a,b)=>b.date.localeCompare(a.date))[0] || null;
  }
  function change(now,before) {
    if (now==null || before==null) return null;
    return {amount:now-before,percent:before===0?null:(now-before)/before*100};
  }
  function snapshot(data) {
    return clean({date:jstDay(data.calculated_at),scope:'simulator',total_value:data.total_value,
      annual_net:data.total_annual_dividend_net,annual_gross:data.total_annual_dividend,
      coverage:data.yield_coverage_pct,source:'計算日時の試算を保存（残高基準日は元データによる）',
      calculated_at:data.calculated_at,method:'dividend-v3'});
  }
  const api={jstDay,validRecord,merge,targetDay,baseline,change,snapshot};
  if(typeof module!=='undefined') module.exports=api;
  root.DividendHistory=api;
  if(typeof document==='undefined')return;
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const yen=n=>n==null?'記録なし':'¥'+Math.round(n).toLocaleString('ja-JP');
  let records=[], storageError='', selectedScope='simulator', chart=null;
  function load() {
    try {
      const raw=localStorage.getItem(KEY);
      if(raw){const parsed=JSON.parse(raw);if(!Array.isArray(parsed)||!parsed.every(validRecord))throw new Error('履歴形式エラー');records=merge([],parsed);}
    } catch(e) {storageError='履歴を読み取れません。ブラウザーの保存設定またはバックアップを確認してください。';}
  }
  function save(next) {
    localStorage.setItem(KEY,JSON.stringify(next));records=next;
  }
  function delta(current,prior,key) {
    const d=change(current[key],prior?.[key]);
    return d ? `${d.amount>=0?'+':'−'}${yen(Math.abs(d.amount))} / ${d.percent==null?'増減率なし（比較元0円）':(d.percent>=0?'+':'')+d.percent.toFixed(2)+'%'}` : '比較データなし';
  }
  function comparison(current,months,label) {
    const prior=baseline(records,current,months);
    const compatible=prior && (current.scope==='mf_securities' || current.method===prior.method && current.coverage===prior.coverage);
    return `<article class="history-compare"><h3>${label}</h3><p>${prior?esc(prior.date)+' → '+esc(current.date):esc(targetDay(current.date,months))+'以前の同月記録なし'}</p>
      <dl><dt>証券評価額の増減</dt><dd>${delta(current,prior,'total_value')}</dd><dt>年間配当試算（税後）の増減</dt><dd>${compatible?delta(current,prior,'annual_net'):'比較条件が揃わないため表示しません'}</dd></dl>
      ${prior && current.coverage!==prior.coverage?'<p>利回り確認率が異なるため、配当試算の単純比較を停止しています。</p>':''}</article>`;
  }
  function render() {
    const host=document.getElementById('historyPanel');if(!host)return;
    const list=records.filter(r=>r.scope===selectedScope),current=list[list.length-1];
    host.innerHTML=`<h2>前月・前年との比較と推移</h2><p>このブラウザーに記録を保存します。同日の再計算は最新値に更新します。別端末との同期はありません。ブラウザーのデータ削除に備え、バックアップしてください。</p>
      <div class="history-actions"><label>比較する範囲 <select id="historyScope">${Object.entries(SCOPES).map(([k,v])=>`<option value="${k}" ${selectedScope===k?'selected':''}>${v}</option>`).join('')}</select></label>
      <button id="historyExport" type="button">履歴をバックアップ</button><label class="history-import">履歴JSONを取り込む<input id="historyImport" type="file" accept=".json,application/json"></label></div>
      <p id="historyStatus" role="status">${esc(storageError)}</p>
      ${current?`<p>最新の保存日：${esc(current.date)}／${list.length}件。${esc(current.source)}。比較元は前月・前年同月の同日以前で最も近い保存日です。</p>
      <div class="history-comparisons">${comparison(current,-1,'前月との比較')}${comparison(current,-12,'前年同月との比較')}</div>
      <div class="history-chart"><canvas id="historyChart" aria-label="保存済み証券評価額と年間配当試算の推移"></canvas></div>
      <details><summary>保存済みの数値を表で見る</summary><div class="table-scroll"><table><thead><tr><th>保存日</th><th>証券評価額</th><th>年間配当試算（税後）</th><th>利回り確認率</th></tr></thead><tbody>${list.map(r=>`<tr><td>${esc(r.date)}</td><td>${yen(r.total_value)}</td><td>${yen(r.annual_net)}</td><td>${r.coverage==null?'記録なし':r.coverage+'%'}</td></tr>`).join('')}</tbody></table></div></details>`:'<p>この範囲の履歴はまだありません。マネーフォワードの履歴JSONや、この画面で保存したバックアップを取り込めます。</p>'}
      <p>評価額の増減には入出金・売買・価格・為替の影響が含まれ、運用損益そのものではありません。配当試算は受取実績ではありません。マネーフォワードの証券残高は国内株・米国株・投資信託の合計で、この画面の取得範囲と分けて比較します。</p>`;
    document.getElementById('historyScope').onchange=e=>{selectedScope=e.target.value;render()};
    document.getElementById('historyExport').onclick=()=>{
      const blob=new Blob([JSON.stringify({format:'dividend-history-v1',records},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
      a.href=url;a.download='dividend-history-'+jstDay(new Date().toISOString())+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    };
    document.getElementById('historyImport').onchange=async e=>{
      try {const file=e.target.files[0];if(!file)return;if(file.size>5e6)throw new Error('5MB以下のファイルを選択してください');
        const payload=JSON.parse((await file.text()).replace(/^\uFEFF/,''));if(payload.format!=='dividend-history-v1')throw new Error('対応する履歴JSONではありません');
        const next=merge(records,payload.records);save(next);storageError='取り込みました。同日・同範囲の既存記録は保持しました。';
        if(payload.records.some(r=>r.scope==='mf_securities'))selectedScope='mf_securities';render();
      }catch(err){document.getElementById('historyStatus').textContent='取り込みできません：'+err.message;}
    };
    if(chart){chart.destroy();chart=null;}
    if(current && typeof Chart!=='undefined')chart=new Chart(document.getElementById('historyChart'),{type:'line',data:{labels:list.map(r=>r.date),datasets:[
      {label:'証券評価額',data:list.map(r=>r.total_value),yAxisID:'assets',borderColor:'#3182ce',tension:0,spanGaps:false},
      {label:'年間配当試算（税後）',data:list.map(r=>r.annual_net),yAxisID:'dividend',borderColor:'#38a169',tension:0,spanGaps:false}
    ]},options:{responsive:true,maintainAspectRatio:false,scales:{assets:{type:'linear',position:'left',title:{display:true,text:'証券評価額（円）'}},dividend:{type:'linear',position:'right',grid:{drawOnChartArea:false},title:{display:true,text:'年間配当試算（円）'}}}}});
  }
  api.attach=function(data){load();if(data.calculated_at && !storageError){try{save(merge(records,[snapshot(data)],true))}catch(e){storageError='履歴を保存できませんでした。空き容量とブラウザーの保存設定を確認してください。'}}render();};
})(typeof window==='undefined'?globalThis:window);
