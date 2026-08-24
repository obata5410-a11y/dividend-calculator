import unittest
from collections import Counter
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


if __name__ == "__main__":
    unittest.main()
