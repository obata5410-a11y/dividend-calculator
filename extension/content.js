// マネーフォワード ME の保有証券データを抽出する

function extractHoldings() {
  const holdings = [];
  const seenBlocks = new Set();

  // ── 戦略1: <table> から抽出 ──────────────────────────────
  for (const table of document.querySelectorAll("table")) {
    const headers = Array.from(table.querySelectorAll("th, thead td")).map(
      (el) => el.textContent.trim()
    );

    const valueIdx = headers.findIndex((h) => h.includes("評価額"));
    if (valueIdx === -1) continue;

    const nameIdx = headers.findIndex(
      (h) => h.includes("銘柄名") || h.includes("銘柄") || h.includes("ファンド名")
    );
    const codeIdx = headers.findIndex(
      (h) => h.includes("コード") || h.includes("証券コード")
    );

    const typeIdx = headers.findIndex(h => /口座区分|預り区分|口座種別/.test(h));
    const accountIdx = typeIdx >= 0 ? typeIdx : headers.findIndex(h => /口座|保有金融機関/.test(h));
    const quantityIdx = headers.findIndex(h => /保有数|数量/.test(h));
    for (const row of table.querySelectorAll("tbody tr")) {
      if (row.closest("table") !== table) continue;
      const cells = Array.from(row.querySelectorAll("td"));
      if (cells.length < 2) continue;

      const rawValue = cells[valueIdx]?.textContent.replace(/[¥,円\s\u00a5]/g, "") || "";
      const value = parseFloat(rawValue);
      if (isNaN(value) || value <= 0) continue;

      const nameCell = cells[nameIdx !== -1 ? nameIdx : 0];
      let name = nameCell?.textContent.replace(/\s+/g, " ").trim() || "";

      // コード: 専用列 → リンクのhref → 銘柄名内の括弧 → 4桁数字
      let code = "";
      if (codeIdx !== -1) {
        code = cells[codeIdx]?.textContent.replace(/\s/g, "") || "";
      }
      if (!code) {
        const link = nameCell?.querySelector("a");
        if (link) {
          const m = link.href.match(/\/(?:stock|etf|fund)s?\/(\w+)/i);
          if (m) code = m[1];
        }
      }
      if (!code) {
        const m = name.match(/[（(](\d{4,5})[）)]/);
        if (m) code = m[1];
      }
      // 銘柄名から余分なコード表記を除去
      name = name.replace(/[（(]\d{4,5}[）)]/g, "").trim();

      if (name) {
        const account = accountIdx >= 0 ? cells[accountIdx]?.textContent.trim() || "" : "";
        const rawQuantity = quantityIdx >= 0 ? cells[quantityIdx]?.textContent.replace(/[,株口\s]/g, "") : "";
        const quantity = rawQuantity && Number.isFinite(Number(rawQuantity)) ? Number(rawQuantity) : null;
        holdings.push({ name, code, value, account, quantity });
      }
    }
  }

  if (holdings.length > 0) return holdings;

  // ── 戦略2: リスト形式のDOM（マネーフォワードのカード型レイアウト）──
  // 評価額を含む要素を親に持つブロックを探す
  const valueEls = document.querySelectorAll(
    '[class*="amount"], [class*="value"], [class*="price"], [class*="valuation"]'
  );

  for (const el of valueEls) {
    const text = el.textContent.replace(/[¥,円\s\u00a5]/g, "");
    const value = parseFloat(text);
    if (isNaN(value) || value < 1000) continue;

    // 親ブロック内から銘柄名・コードを探す
    const block = el.closest("li, [class*='item'], [class*='stock'], [class*='fund'], article, section");
    if (!block) continue;

    const nameEl = block.querySelector(
      '[class*="name"], [class*="title"], a, h3, h4'
    );
    if (!nameEl) continue;

    let name = nameEl.textContent.replace(/\s+/g, " ").trim();
    let code = "";

    const linkEl = block.querySelector("a");
    if (linkEl) {
      const m = linkEl.href.match(/\/(?:stock|etf|fund)s?\/(\w+)/i);
      if (m) code = m[1];
    }
    const codeMatch = block.textContent.match(/[（(](\d{4,5})[）)]/);
    if (!code && codeMatch) code = codeMatch[1];
    name = name.replace(/[（(]\d{4,5}[）)]/g, "").trim();

    if (!seenBlocks.has(block) && name && name.length < 60) {
      seenBlocks.add(block);
      holdings.push({ name, code, value });
    }
  }

  return holdings;
}

chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
  if (request.action === "extract") {
    const holdings = extractHoldings();
    sendResponse({ holdings, count: holdings.length, url: location.href });
  }
  return true;
});
