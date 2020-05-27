# -*- coding: utf-8 -*-

"""
Unit tests for the filterdict module.
"""

import unittest as ut
import filterdict as fd


_meta = fd.FilterDictMeta
_d = _meta._dics


class DummyDict(fd.FilterDict):
    @staticmethod
    def keycheck(k):
        return k != 42

    @staticmethod
    def counter_eg():
        return 42


class DummyClass:
    pass


class TestValidUse(ut.TestCase):
    def setUp(self):
        def a():
            return (
                [
                    (1, 1),
                    ('2', 2),
                    ('__getitem__', 3)
                ],
                dict(a=4, b='5'),
                DummyClass(),
                (
                    ('x', 'y'),
                    (6, '6')
                ),
                'ignored',
                map(lambda x, y: (x, y), 'cde', range(7, 10)),
                {(0, 0): 'double zero'}
            )
        self.kw = kw = dict(p1=11, p2=22)
        self.dummydic = DummyDict(*a(), **kw)

        sk = '2', 'a', 'b', 'x', 'c', 'd', 'e', 'p1', 'p2'

        # expected valid keys for StrDict
        self.strdic_keys = frozenset(sk)
        self.strdic = fd.StrDict(*a(), **kw)

        # expected valid keys for NsDict
        self.nsdic_keys = frozenset(sk[1:])
        self.nsdic = fd.NsDict(*a(), **kw)

        self.nnsdic = fd.NestedNsDict(*a(), **kw)

        self.dictdic = d = {}
        for e in a():
            try:
                d.update(e)
            except (TypeError, ValueError):
                pass
        d.update(**kw)
        for r in _meta._reserved:
            d.pop(r, None)

    def test_init_DummyDict(self):
        self.assertEqual(self.dictdic, _d[self.dummydic])

    def test_init_StrDict(self):
        self.assertEqual(self.strdic.keys(), set(self.strdic_keys))

    def test_init_NsDict(self):
        self.assertEqual(self.nsdic.keys(), set(self.nsdic_keys))

    def test_Dummy_setitem(self):
        d = self.dummydic
        d['a'] = 42
        self.assertEqual(d['a'], 42)

    def test_Dummy_getattr(self):
        dd = self.dummydic
        self.assertEqual(dd.a, 4)
        self.assertEqual(getattr(dd, 'a', None), 4)

    def test_Dummy_setattr(self):
        d = self.dummydic
        d.a = 42
        self.assertEqual(d['a'], 42)

    def test_Dummy_reserved(self):
        for r in _meta._reserved:
            self.assertNotIn(r, self.dummydic)

    def test_Str_lt_Dummy(self):
        self.assertLess(self.strdic.keys(), self.dummydic.keys())

    def test_Ns_lt_Str(self):
        self.assertLess(self.nsdic.keys(), self.strdic.keys())

    def test_Ns_eq_Nns(self):
        self.assertEqual(self.nsdic, self.nnsdic)

    def test_Ns_missing_attr_get(self):
        with self.assertRaises(AttributeError):
            aa = self.nsdic.aa

    def test_Nns_missing_attr_get(self):
        self.assertEqual(self.nnsdic.aa, {})

    def test_Nns_chained_attr_set(self):
        self.assertEqual(self.nsdic.aa.bb.cc, 42)

    def test_Nns_parent(self):
        d = self.dummydic
        d.aa.bb.cc.dd = 42
        self.assertEqual(d.aa.bb.cc.parent, d.aa.bb)
        self.assertEqual(d.aa.bb.cc.parent['cc'], dict(dd=42))
        self.assertIsNone(d.parent)
        self.assertIsNone(d.parent.parent)


if __name__ == '__main__':
    ut.main()
