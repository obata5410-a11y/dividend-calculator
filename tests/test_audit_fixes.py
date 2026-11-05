import unittest
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import patch
import app


class AuditFixes(unittest.TestCase):
    def run_holdings(self, items, data=None, fx=150, distributions=None):
        with (patch.object(app, 'load_nisa_lot_counts', return_value=Counter({'2914': 1})),
              patch.object(app, 'load_ticker_aliases', return_value={}),
              patch.object(app, 'load_yield_overrides', return_value={}),
              patch.object(app, 'load_dividend_evidence', return_value={}),
              patch.object(app, 'load_distribution_history', return_value=distributions or {}),
              patch.object(app, 'fetch_ticker_data', return_value=data or {'div_yield': .12}),
              patch.object(app, 'fetch_usd_jpy_rate', return_value=fx),
              patch.object(app.time, 'sleep')):
            return app.process_holdings(items)

    def test_unknown_account_is_order_independent_and_explicit(self):
        items = [app.HoldingItem(name='JT', code='2914', value=v) for v in (100000, 200000)]
        a, b = self.run_holdings(items), self.run_holdings(items[::-1])
        self.assertEqual(a['total_annual_dividend_net'], b['total_annual_dividend_net'])
        self.assertEqual(a['unknown_account_count'], 2)
        self.assertEqual(a['holdings'][0]['nisa_status'], 'unknown')
        self.assertEqual(a['holdings'][0]['nisa_value'], 0)

    def test_only_account_type_establishes_tax_status(self):
        for text in ['', 'SBI証券', '楽天証券']:
            self.assertEqual(app.account_tax_status(text), 'unknown')
        for text in ['NISA成長投資枠', 'ＮＩＳＡ', '非課税']:
            self.assertEqual(app.account_tax_status(text), 'nisa')
        for text in ['特定口座', '一般口座', '課税']:
            self.assertEqual(app.account_tax_status(text), 'taxable')

    def test_effective_tax_uses_dividends_not_valuation(self):
        items = [app.HoldingItem(name='TEST', code='TEST', value=100000, quantity=10, account='NISA'),
                 app.HoldingItem(name='TEST', code='TEST', value=100000, quantity=30, account='特定')]
        h = self.run_holdings(items, {'div_yield': .04, 'annual_dividend_per_share': 4, 'currency': 'USD'})['holdings'][0]
        self.assertAlmostEqual(h['tax_rate'], 23.713, places=3)
        self.assertEqual(h['annual_dividend'], 24000)
        self.assertEqual(h['annual_dividend_net'], 18309)

    def test_fund_uses_units_and_not_valuation(self):
        dist = {'FUND': {'annual_distribution': 570, 'unit': 10000}}
        results = [self.run_holdings([app.HoldingItem(name='FUND', value=v, quantity=100390, account='NISA')], distributions=dist)['holdings'][0] for v in (170411, 200000)]
        self.assertEqual(results[0]['annual_dividend'], 5722)
        self.assertEqual(results[0]['annual_dividend_net'], results[1]['annual_dividend_net'])
        self.assertEqual(results[0]['calculation_method'], 'fund_quantity_ttm')

    def test_missing_fund_quantity_is_not_valued_with_old_fixed_yield(self):
        name = 'SBI日本高配当株式(分配)ファンド(年4回決算型)'
        self.assertNotIn(app._normalize_config_key(name), app.load_yield_overrides())
        h = self.run_holdings([app.HoldingItem(name='FUND', value=170411)], data={'div_yield': None}, distributions={'FUND': {'annual_distribution': 570, 'unit': 10000}})['holdings'][0]
        self.assertIsNone(h['annual_dividend'])

    def test_expired_fx_not_used_and_us_amount_excluded(self):
        with (patch.object(app, '_fx_cache', {'rate': 100, 'ts': 1}),
              patch.object(app.time, 'time', return_value=2000000000),
              patch.object(app._HTTP, 'get', side_effect=RuntimeError('offline'))):
            self.assertIsNone(app.fetch_usd_jpy_rate())
        h = self.run_holdings([app.HoldingItem(name='TEST', code='TEST', value=100000, quantity=10)],
            {'div_yield': .04, 'annual_dividend_per_share': 4, 'currency': 'USD'}, fx=None)['holdings'][0]
        self.assertIsNone(h['annual_dividend'])
        self.assertEqual(h['yield_source'], 'fx_unavailable')

    def test_successful_fx_and_fresh_cache(self):
        response = unittest.mock.Mock(status_code=200)
        response.json.return_value = {'chart': {'result': [{'meta': {'regularMarketPrice': 150}}]}}
        with (patch.object(app, '_fx_cache', {'rate': 100, 'ts': 1}),
              patch.object(app.time, 'time', return_value=2000000000),
              patch.object(app._HTTP, 'get', return_value=response) as fetch):
            self.assertEqual(app.fetch_usd_jpy_rate(), 150)
            self.assertEqual(app.fetch_usd_jpy_rate(), 150)
            self.assertEqual(fetch.call_count, 1)

    def test_distribution_evidence_expires(self):
        with patch.object(app, 'datetime', wraps=datetime) as clock:
            clock.now.return_value = datetime(2027, 1, 12, tzinfo=timezone.utc)
            self.assertEqual(app.load_distribution_history(), {})


if __name__ == '__main__':
    unittest.main()
