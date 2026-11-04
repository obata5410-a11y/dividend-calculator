const fs=require('fs'),vm=require('vm'),assert=require('assert');
(async()=>{
 const nodes=new Map(),local=new Map(),session=new Map(),calls=[];
 const doc={getElementById(id){if(!nodes.has(id))nodes.set(id,{innerHTML:'',value:'',parentElement:{style:{}},classList:{toggle(){}}});return nodes.get(id)}};
 let remote=[{date:'2026-10-05',scope:'mf_securities',total_value:100,annual_net:null,annual_gross:null,coverage:null,method:'mf-balances-v1',source:'MF'}],fail=false;
 const fetch=async(path,opts={})=>{
  calls.push({path,opts});
  if(path.endsWith('/status'))return {ok:true,json:async()=>({configured:true})};
  assert.equal(opts.headers.Authorization,'Bearer '+'x'.repeat(40));
  if(fail)return {ok:false,json:async()=>({detail:'temporary failure'})};
  if(opts.method==='POST'){const data=JSON.parse(opts.body);remote.push(...data.records);return {ok:true,json:async()=>({created:data.records.length})};}
  return {ok:true,json:async()=>({records:remote})};
 };
 const ctx={document:doc,localStorage:{getItem:k=>local.get(k)||null,setItem:(k,v)=>local.set(k,v)},sessionStorage:{getItem:k=>session.get(k)||null,setItem:(k,v)=>session.set(k,v),removeItem:k=>session.delete(k)},window:{},fetch,Chart:function(){this.destroy=()=>{}},setTimeout};
 vm.createContext(ctx);vm.runInContext(fs.readFileSync('static/history.js','utf8'),ctx);
 ctx.window.DividendHistory.attach({calculated_at:'2026-11-04T16:00:00Z',total_value:120,total_annual_dividend_net:5,total_annual_dividend:6,yield_coverage_pct:100});
 await new Promise(setImmediate);
 doc.getElementById('notionAccess').value='x'.repeat(40);await nodes.get('notionConnect').onclick();
 assert(nodes.get('historyPanel').innerHTML.includes('Notionに保存・読戻し確認済み'));
 assert.equal(remote.length,2);assert.equal(session.get('dividend-notion-access'),'x'.repeat(40));
 assert(![...local.values()].join('').includes('x'.repeat(40)));
 const saved=local.get('dividend-history-v1');fail=true;
 await nodes.get('notionConnect').onclick();
 assert(nodes.get('notionStatus').textContent.includes('同期未完了'));
 assert.equal(local.get('dividend-history-v1'),saved);
 nodes.get('notionDisconnect').onclick();assert.equal(session.size,0);assert.equal(remote.length,2);
 console.log('Cloud UI: auth, read/write/readback, secret isolation, failure preservation, disconnect PASS');
})().catch(e=>{console.error(e);process.exitCode=1});
