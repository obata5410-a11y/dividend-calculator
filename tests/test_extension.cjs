const fs=require('fs'),vm=require('vm'),assert=require('assert');
const header=t=>({textContent:t});let table;
const row=account=>({closest:()=>table,querySelectorAll:()=>['TEST','Same holding','10,000','10',account].map(text=>({textContent:text,querySelector:()=>null}))});
table={querySelectorAll:selector=>selector==='th, thead td'?['コード','銘柄名','評価額','保有数','口座'].map(header):[row('NISA'),row('特定')]};
const ctx={document:{querySelectorAll:()=>[table]},chrome:{runtime:{onMessage:{addListener(){}}}}};vm.createContext(ctx);vm.runInContext(fs.readFileSync('extension/content.js','utf8'),ctx);
const result=ctx.extractHoldings();assert.equal(result.length,2);assert.equal(result[0].quantity,10);assert.equal(result[1].account,'特定');console.log('Extension: equal-value holdings in separate accounts retained with quantity PASS');
table={querySelectorAll:selector=>selector==='th, thead td'?['コード','銘柄名','評価額','保有数','保有金融機関','口座区分'].map(header):[
{closest:()=>table,querySelectorAll:()=>['TEST','Name','10,000','10','SBI証券','NISA'].map(text=>({textContent:text,querySelector:()=>null}))}]};
assert.equal(ctx.extractHoldings()[0].account,'NISA');
