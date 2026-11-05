const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes=new Map(),storage=new Map();
const seed=[{date:'2026-10-05',scope:'simulator',total_value:100000,annual_net:3600,coverage:100,method:'dividend-v3'},{date:'2025-11-05',scope:'simulator',total_value:80000,annual_net:3000,coverage:100,method:'dividend-v3'}];
storage.set('dividend-history-v1',JSON.stringify(seed));
const document={getElementById(id){if(!nodes.has(id))nodes.set(id,{innerHTML:'',parentElement:{style:{}},classList:{toggle(){}}});return nodes.get(id)}};
const ctx={document,localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},window:{},location:{search:''},URLSearchParams,Chart:function(el,cfg){this.data=cfg.data;this.update=()=>{};this.destroy=()=>{}},setTimeout};
vm.createContext(ctx);vm.runInContext(fs.readFileSync('static/history.js','utf8'),ctx);
const script=fs.readFileSync('static/result.html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];vm.runInContext(script,ctx);
ctx.render({calculated_at:'2026-11-04T16:00:00Z',holdings:[{name:'TEST',ticker:'TEST',code:'TEST',value:120000,div_yield:4,annual_dividend:4800,annual_dividend_net:4320,monthly_dividend:400,monthly_dividend_net:360,tax_rate:10,nisa_status:'all',calculation_method:'quantity_ttm'}],total_value:120000,total_annual_dividend:4800,total_annual_dividend_net:4320,total_monthly_dividend:400,total_monthly_dividend_net:360,portfolio_dividend_yield:4,yield_coverage_pct:100});
assert(nodes.get('historyPanel').innerHTML.includes('+¥20,000 / +20.00%'));
assert(nodes.get('historyPanel').innerHTML.includes('+¥40,000 / +50.00%'));
assert(nodes.get('app').innerHTML.includes('数量×過去1年配当'));
assert.equal(JSON.parse(storage.get('dividend-history-v1')).length,3);
ctx.switchBarMode('annual');assert.equal(ctx.window._barChart.data.datasets[0].data[0],4800);
console.log('Result rendering, automatic history save, month/year amounts and percentages, annual chart toggle PASS');
ctx.render({calculation_version:'dividend-v4',calculated_at:'2026-11-05T16:00:00Z',unknown_account_count:1,
holdings:[{name:'FUND',value:170411,div_yield:3.36,annual_dividend:5722,annual_dividend_net:4560,
monthly_dividend:477,monthly_dividend_net:380,tax_rate:20.315,nisa_status:'unknown',calculation_method:'fund_quantity_ttm',
distribution_evidence:{publisher:'Issuer',period_start:'2025-11-06',period_end:'2026-11-05',unit:10000,annual_distribution:570,review_after:'2027-01-12',source_url:'https://example.com/distributions'}}],
total_value:170411,total_annual_dividend:5722,total_annual_dividend_net:4560,total_monthly_dividend:477,total_monthly_dividend_net:380,yield_coverage_pct:100});
assert(nodes.get('app').innerHTML.includes('NISAの推測割当は行いません'));
assert(nodes.get('app').innerHTML.includes('口座区分不明あり'));
assert(nodes.get('app').innerHTML.includes('保有口数÷1万口'));
assert(nodes.get('app').innerHTML.includes('https://example.com/distributions'));
assert.equal(JSON.parse(storage.get('dividend-history-v1')).at(-1).method,'dividend-v4');
