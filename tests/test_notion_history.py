import os
import unittest
from unittest.mock import Mock, patch
from datetime import date
from fastapi import HTTPException
from pydantic import ValidationError
import notion_history as history


class NotionHistoryTests(unittest.TestCase):
    def sample(self, **kwargs):
        return history.Snapshot(date='2026-10-05', scope='mf_securities', total_value=100,
            source='Source', method='mf-balances-v1', **kwargs)

    def store(self):
        with patch.dict(os.environ, {'NOTION_TOKEN':'test-notion-token','NOTION_DATA_SOURCE_ID':'test-data-source'}):
            return history.NotionStore()

    def test_not_configured_without_all_three_server_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(history.configured())
            with self.assertRaises(HTTPException) as err:
                history.authorize(None)
            self.assertEqual(err.exception.status_code,401)

    def test_auth_denies_wrong_key_even_when_configured(self):
        with patch.dict(os.environ, {'NOTION_TOKEN':'test','NOTION_DATA_SOURCE_ID':'test', 'DIVIDEND_HISTORY_ACCESS_KEY':'x'*40}):
            with self.assertRaises(HTTPException) as err: history.authorize('Bearer wrong')
            self.assertEqual(err.exception.status_code,401)
            history.authorize('Bearer '+'x'*40)

    def test_snapshot_validation_rejects_invalid_money_and_dates(self):
        for amount in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValidationError): history.Snapshot(date='2026-10-05',scope='simulator',total_value=amount)
        with self.assertRaises(ValidationError): history.Snapshot(date='2999-01-01',scope='simulator',total_value=1)
        with self.assertRaises(ValidationError): self.sample(calculated_at='2026-10-05T12:00:00')

    def test_deduplication_ignores_source_wording_but_keeps_changed_figures(self):
        a=self.sample(); b=a.model_copy(update={'source':'Other wording'})
        self.assertEqual(history.snapshot_key(a),history.snapshot_key(b))
        self.assertNotEqual(history.snapshot_key(a),history.snapshot_key(a.model_copy(update={'total_value':101.0})))

    def test_properties_preserve_missing_dividends_as_null(self):
        p=history.NotionStore.properties(self.sample())
        self.assertIsNone(p['AnnualNet']['number'])
        self.assertEqual(p['TotalValue']['number'],100)
        self.assertEqual(p['Date']['date']['start'],'2026-10-05')

    def test_duplicate_lookup_prevents_repeat_write(self):
        store=self.store();store.request=Mock(return_value={'results':[{'id':'existing'}],'has_more':False})
        self.assertFalse(store.append(self.sample()))
        self.assertEqual(store.request.call_count,1)

    def test_changed_record_appends_without_updating_old_page(self):
        store=self.store();store.request=Mock(side_effect=[{'results':[],'has_more':False},{'id':'new'}])
        self.assertTrue(store.append(self.sample()))
        self.assertEqual(store.request.call_args.args[0],'pages')
        self.assertEqual(store.request.call_args.args[1]['parent']['data_source_id'],'test-data-source')

    def test_pagination_reads_every_page(self):
        store=self.store();store.request=Mock(side_effect=[{'results':[1],'has_more':True,'next_cursor':'next'}, {'results':[2],'has_more':False}])
        self.assertEqual(store.pages(),[1,2])
        self.assertEqual(store.request.call_args.args[1]['start_cursor'],'next')

    def test_write_timeout_is_not_retried_and_secret_is_not_returned(self):
        store=self.store()
        with patch.object(history.requests,'post',side_effect=history.requests.Timeout('sensitive upstream error')) as post:
            with self.assertRaises(HTTPException) as err: store.request('pages',{})
        self.assertEqual(post.call_count,1)
        self.assertEqual(err.exception.status_code,504)
        self.assertNotIn('sensitive',err.exception.detail)
        self.assertNotIn('test-notion-token',err.exception.detail)

    def test_read_returns_newest_revision_and_keeps_nulls(self):
        store=self.store()
        def page(value,stamp):
            return {'properties':store.properties(self.sample().model_copy(update={'total_value':value})), 'created_time':stamp}
        store.pages=Mock(return_value=[page(100,'2026-10-05T00:00:00Z'),page(120,'2026-10-05T01:00:00Z')])
        records=store.records()
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['total_value'],120)
        self.assertIsNone(records[0]['annual_net'])

    def test_partial_query_failure_never_returns_partial_records(self):
        store=self.store(); store.request=Mock(side_effect=[{'results':[1],'has_more':True,'next_cursor':'next'},HTTPException(status_code=502)])
        with self.assertRaises(HTTPException): store.pages()

if __name__=='__main__': unittest.main()
