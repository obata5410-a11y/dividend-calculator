import unittest
from datetime import date, datetime, timezone
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import app


class ProcessHoldingsTests(unittest.TestCase):
    def run_process(self, items, *, nisa=None, aliases=None, overrides=None, yields=None):
        nisa = Counter(nisa or {})
        aliases = aliases or {}
        overrides = overrides or {}
        yields = yields or {}

        def fake_fetch(ticker):
            value = yields.get(ticker)
            return {"div_yield": value, "name": None}

        with (
            patch.object(app, "load_nisa_lot_counts", return_value=nisa),
            patch.object(app, "load_ticker_aliases", return_value=aliases),
            patch.object(app, "load_yield_overrides", return_value=overrides),
            patch.object(app, "fetch_ticker_data", side_effect=fake_fetch),
            patch.object(app.time, "sleep"),
            patch.object(app, "fetch_usd_jpy_rate", return_value=None),
        ):
            return app.process_holdings(items)

    def test_mixed_nisa_and_taxable_lots_use_effective_tax_rate(self):
        result = self.run_process(
            [
                app.HoldingItem(name="JT", code="2914", value=100_000),
                app.HoldingItem(name="JT", code="2914", value=200_000),
            ],
            nisa={"2914": 1},
            yields={"2914.T": 0.12},
        )

        holding = result["holdings"][0]
        self.assertEqual(holding["value"], 300_000)
        self.assertEqual(holding["nisa_status"], "partial")
        self.assertEqual(holding["nisa_value"], 100_000)
        self.assertAlmostEqual(holding["tax_rate"], 13.543, places=3)
        self.assertEqual(holding["monthly_dividend"], 3_000)
        self.assertEqual(holding["monthly_dividend_net"], 2_594)
        self.assertEqual(result["portfolio_dividend_yield"], 12.0)

    def test_us_nisa_keeps_ten_percent_source_tax(self):
        result = self.run_process(
            [app.HoldingItem(name="SHY", code="SHY", value=100_000)],
            nisa={"SHY": 1},
            yields={"SHY": 0.12},
        )

        holding = result["holdings"][0]
        self.assertEqual(holding["nisa_status"], "all")
        self.assertEqual(holding["tax_rate"], 10.0)
        self.assertEqual(holding["monthly_dividend"], 1_000)
        self.assertEqual(holding["monthly_dividend_net"], 900)

    def test_explicit_account_name_takes_priority_over_legacy_nisa_list(self):
        result = self.run_process(
            [app.HoldingItem(name="JT", code="2914", value=100_000, account="楽天証券")],
            nisa={"2914": 1},
            yields={"2914.T": 0.12},
        )

        holding = result["holdings"][0]
        self.assertEqual(holding["nisa_status"], "none")
        self.assertEqual(holding["tax_rate"], 20.315)

    def test_alias_and_yield_override_cover_code_less_holdings(self):
        result = self.run_process(
            [
                app.HoldingItem(name="CSCO ESPP", value=100_000),
                app.HoldingItem(
                    name="SBI日本高配当株式(分配)ファンド(年4回決算型)",
                    value=100_000,
                ),
            ],
            aliases={"CSCO ESPP": "CSCO"},
            overrides={"SBI日本高配当株式(分配)ファンド(年4回決算型)": 0.04},
            yields={"CSCO": 0.03},
        )

        csco, fund = result["holdings"]
        self.assertEqual(csco["code"], "CSCO")
        self.assertEqual(csco["ticker"], "CSCO")
        self.assertEqual(csco["div_yield"], 3.0)
        self.assertEqual(fund["yield_source"], "override")
        self.assertEqual(fund["div_yield"], 4.0)
        self.assertEqual(result["portfolio_dividend_yield"], 3.5)
        self.assertEqual(result["yield_coverage_pct"], 100.0)

    def test_code_less_fund_can_be_marked_nisa_by_name(self):
        name = "SBI日本高配当株式(分配)ファンド(年4回決算型)"
        result = self.run_process(
            [
                app.HoldingItem(name=name, value=10_000),
                app.HoldingItem(name=name, value=20_000),
            ],
            nisa={name.upper(): 2},
            overrides={name.upper(): 0.04},
        )

        holding = result["holdings"][0]
        self.assertEqual(holding["nisa_status"], "all")
        self.assertEqual(holding["nisa_value"], 30_000)
        self.assertEqual(holding["tax_rate"], 0.0)

    def test_domestic_fund_tax_category_matches_nisa_calculation(self):
        result = self.run_process(
            [app.HoldingItem(name="国内投信", value=10000, account="NISA")],
            overrides={"国内投信": 0.04},
        )
        holding = result["holdings"][0]
        self.assertEqual(holding["tax_category"], "JP")
        self.assertEqual(holding["tax_rate"], 0)
        self.assertEqual(holding["annual_dividend_net"], 400)

    def test_annual_totals_do_not_multiply_rounded_months(self):
        result = self.run_process(
            [app.HoldingItem(name="JT", code="2914", value=10000)],
            yields={"2914.T": 0.04},
        )
        self.assertEqual(result["total_annual_dividend"], 400)
        self.assertEqual(result["total_annual_dividend_net"], 319)

    def test_confirmed_zero_is_distinct_from_missing(self):
        result = self.run_process(
            [app.HoldingItem(name="zero", code="ZERO", value=10000),
             app.HoldingItem(name="missing", code="MISSING", value=10000)],
            yields={"ZERO": 0.0},
        )
        zero, missing = result["holdings"]
        self.assertEqual(zero["annual_dividend"], 0)
        self.assertIsNone(missing["annual_dividend"])
        self.assertEqual(result["yield_coverage_pct"], 50)
        self.assertEqual(result["missing_yield_count"], 1)


class DividendEvidenceTests(unittest.TestCase):
    run_process = ProcessHoldingsTests.run_process
    def setUp(self):
        original = app.find_dividend_evidence
        self.frozen = patch.object(app, "find_dividend_evidence",
            side_effect=lambda *args, **kwargs: original(*args, today=date(2026, 11, 5)))
        self.frozen.start()
        self.addCleanup(self.frozen.stop)

    def test_all_twelve_missing_holdings_have_sourced_zero_history(self):
        records = app.load_dividend_evidence()
        self.assertEqual(len(records), 12)
        items = [app.HoldingItem(name=r["key"], code=r["key"] if r["key"] in ("2516", "7201", "GLD") else "", value=10000) for r in records.values()]
        result = self.run_process(items)
        self.assertEqual(result["missing_yield_count"], 0)
        self.assertEqual(result["confirmed_zero_count"], 12)
        self.assertEqual(result["yield_coverage_pct"], 100)
        self.assertEqual(result["total_annual_dividend_net"], 0)
        for h in result["holdings"]:
            self.assertEqual(h["yield_source"], "verified_history")
            self.assertTrue(h["dividend_evidence"]["source_url"].startswith("https://"))

    def test_new_positive_market_dividend_takes_priority(self):
        result = self.run_process([app.HoldingItem(name="日産", code="7201", value=10000)], yields={"7201.T": 0.02})
        h = result["holdings"][0]
        self.assertEqual(h["div_yield"], 2)
        self.assertEqual(h["yield_source"], "market")
        self.assertIsNone(h["dividend_evidence"])

    def test_monthly_distribution_variant_and_unknown_code_remain_missing(self):
        result = self.run_process([
            app.HoldingItem(name="iFreeNEXT FANG+インデックス(毎月決算/予想分配金提示型)", value=10000),
            app.HoldingItem(name="iFreeNEXT FANG+インデックス", code="UNKNOWN", value=10000),
        ])
        self.assertEqual(result["missing_yield_count"], 2)

    def test_fullwidth_punctuation_matches_exact_fund(self):
        result = self.run_process([app.HoldingItem(name="ｅＭＡＸＩＳ Ｓｌｉｍ 全世界株式（オール・カントリー）", value=10000)])
        self.assertEqual(result["confirmed_zero_count"], 1)

    def test_review_date_uses_japan_day_when_server_is_previous_utc_day(self):
        original = self.frozen.temp_original
        records = app.load_dividend_evidence()
        with patch.object(app, "datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc)
            self.assertIsNotNone(original("GLD", "GLD", "", records))

    def test_expired_and_future_dated_evidence_is_not_used(self):
        # Access original function because setUp freezes the date only for integration tests.
        original = self.frozen.temp_original
        records = app.load_dividend_evidence()
        self.assertIsNone(original("2516", "2516.T", "", records, today=date(2027, 2, 3)))
        self.assertIsNone(original("2516", "2516.T", "", records, today=date(2026, 11, 4)))


class ResultTemplateTests(unittest.TestCase):
    def test_after_tax_annual_total_is_a_standalone_summary_category(self):
        template = (Path(__file__).parents[1] / "static" / "result.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("年間配当合計（税引き後）", template)
        self.assertIn("${fmt(data.total_annual_dividend_net)}", template)
        self.assertNotIn("年次配当（税引き後）", template)

    def test_tax_flow_shows_burden_and_take_home_rates(self):
        template = (Path(__file__).parents[1] / "static" / "result.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("年間税負担（概算）", template)
        self.assertIn("annualGross - annualNet", template)
        self.assertIn("手取り率 ${takeHomeRate}%", template)
        self.assertIn("税負担率 ${taxBurdenRate}%", template)
        self.assertIn('class="tax-bar-net"', template)
        self.assertIn('class="tax-bar-tax"', template)


if __name__ == "__main__":
    unittest.main()

