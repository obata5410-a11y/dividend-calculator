const API_BASE = "https://dividend-calculator-6rsz.onrender.com";

const statusEl = document.getElementById("status");
const serverEl = document.getElementById("serverStatus");
const btn      = document.getElementById("btn");

function setStatus(msg, type = "") {
  statusEl.textContent = msg;
  statusEl.className = "status " + type;
}

function setServer(ok) {
  if (ok) {
    serverEl.className = "server-badge ok";
    serverEl.textContent = "● サーバー 起動中";
  } else {
    serverEl.className = "server-badge ng";
    serverEl.textContent = "● サーバー 停止中";
  }
}

// ── サーバー死活確認（1回） ────────────────────────────────
async function checkServer() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(10000) });
    return res.ok;
  } catch {
    return false;
  }
}

// ── サーバー起動待機（リトライ付き） ──────────────────────
// maxMs: 最大待機時間(ms)、intervalMs: 再試行間隔(ms)
async function waitForServer({ maxMs = 60000, intervalMs = 10000 } = {}) {
  const deadline = Date.now() + maxMs;
  let attempt = 0;

  while (true) {
    attempt++;
    const remaining = Math.ceil((deadline - Date.now()) / 1000);
    setStatus(`サーバーに接続中… (試行 ${attempt} 回目 / タイムアウトまで残り約 ${remaining} 秒)`, "warn");

    if (await checkServer()) return true;

    // デッドライン超過またはギリギリなら即終了
    if (Date.now() + intervalMs >= deadline) break;

    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return false;
}

// ── 起動時チェック ─────────────────────────────────────────
(async () => {
  const alive = await waitForServer();
  setServer(alive);

  if (!alive) {
    setStatus("サーバーが 1 分以内に応答しませんでした。時間をおいて再度お試しください。", "err");
    btn.disabled = true;
    return;
  }

  // マネーフォワードのページ確認
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url?.includes("moneyforward.com")) {
    setStatus("マネーフォワード ME のページを開いてください");
    btn.disabled = true;
    return;
  }

  setStatus("マネーフォワード ME を検出しました。銘柄一覧が表示された状態でボタンを押してください", "ok");
  btn.disabled = false;
})();

// ── Chrome Built-in AI (Gemini Nano) ──────────────────────
async function checkAIAvailability() {
  if (typeof LanguageModel === "undefined") return false;
  try {
    const avail = await LanguageModel.availability();
    return avail !== "unavailable";
  } catch {
    return false;
  }
}

function buildPortfolioPrompt(holdings, totals) {
  const fmt = (n) => Math.round(n).toLocaleString("ja-JP");
  const lines = holdings
    .filter((h) => h.div_yield > 0)
    .map((h) => `・${h.name}（${h.code}）: 評価額${fmt(h.value)}円、利回り${Number(h.div_yield).toFixed(2)}%、月次配当（税後）${fmt(h.monthly_dividend_net)}円`)
    .join("\n");

  return `以下は私の配当株ポートフォリオです。

## 保有銘柄
${lines || "（データなし）"}

## 合計
・総評価額: ${fmt(totals.total_value)}円
・月次配当（税後）: ${fmt(totals.total_monthly_dividend_net)}円
・年間配当（税後）: ${fmt(totals.total_annual_dividend_net)}円

このポートフォリオの特徴・強み・改善点を3〜5点、日本語の箇条書きで教えてください。`;
}

async function showAISection(data) {
  const aiSection = document.getElementById("ai-section");
  const aiBtn = document.getElementById("ai-analyze-btn");
  const aiStatus = document.getElementById("ai-status");
  const aiResult = document.getElementById("ai-result");

  const available = await checkAIAvailability();
  if (!available) {
    aiStatus.textContent = "※ AI分析はChrome 127以降 + Gemini Nanoが必要です";
    aiStatus.style.display = "block";
    aiBtn.disabled = true;
  }
  aiSection.style.display = "block";

  aiBtn.addEventListener("click", async () => {
    aiBtn.disabled = true;
    aiBtn.innerHTML = '<span class="spinner"></span>Gemini Nano 分析中…';
    aiStatus.style.display = "none";
    aiResult.style.display = "none";

    try {
      const session = await LanguageModel.create({
        systemPrompt: "あなたは日本人投資家向けの配当ポートフォリオアドバイザーです。簡潔・具体的な日本語でアドバイスしてください。",
      });
      const text = await session.prompt(buildPortfolioPrompt(data.holdings, data));
      session.destroy();
      aiResult.textContent = text;
      aiResult.style.display = "block";
    } catch (e) {
      aiStatus.textContent = "AI分析エラー: " + e.message;
      aiStatus.style.display = "block";
    } finally {
      aiBtn.disabled = false;
      aiBtn.textContent = "✨ AIでポートフォリオ分析";
    }
  });
}

// ── データ取得ボタン ───────────────────────────────────────
btn.addEventListener("click", async () => {
  const alive = await waitForServer();
  if (!alive) {
    setStatus("サーバーが 1 分以内に応答しませんでした。時間をおいて再度お試しください。", "err");
    setServer(false);
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>取得中…';
  setStatus("ページからデータを抽出しています…");

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const response = await chrome.tabs.sendMessage(tab.id, { action: "extract" });

    if (!response || response.count === 0) {
      setStatus("銘柄データが見つかりませんでした。株式（現物）などの個別銘柄リストが表示されているページで試してください", "err");
      return;
    }

    setStatus(`${response.count} 銘柄を検出。配当データを取得中…（銘柄数により1〜2分かかります）`);

    const res = await fetch(`${API_BASE}/api/calculate-from-json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ holdings: response.holdings }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `サーバーエラー (${res.status})`);
    }

    const data = await res.json();
    chrome.tabs.create({ url: `${API_BASE}/result?s=${data.session_id}` });
    setStatus("完了！結果を新しいタブで開きました ✓", "ok");
    await showAISection(data);

  } catch (e) {
    if (e.message.includes("Could not establish connection") || e.message.includes("Receiving end does not exist")) {
      setStatus("ページを再読み込み（F5）してから再度お試しください", "err");
    } else {
      setStatus("エラー: " + e.message, "err");
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "データを取得して試算する";
  }
});
