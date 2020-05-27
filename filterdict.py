# -*- coding: utf-8 -*-

"""
.. module:: filterdict
   :platform: Unix, Windows
   :synopsis: Dictionaries with key constraints.

.. moduleauthor:: Greg Sotiropoulos <greg.sotiropoulos@gmail.com>

Dictionary Abstract Base Class (ABC), ie one that derives from
``collections.abc.MutableMapping``. This ABC provides a systematic way for
subclasses to impose constraints on the kinds of keys that should be
considered valid (beyond the default constraint of a key being hashable and
immutable).

key/value pairs where the key violates the constraints cannot be inserted in
the dictionary (unless one really goes out of their way to do so). Constraints
are specified via the ``keycheck`` method, although a second method
``counter_eg`` is also required (see relevant documentation for more on this).
In cases of invalid keys (any ``k`` for which ``dic.keycheck(k) == False``),
use of ``dic[k] = v`` (or ``dic.setdefault(k, v)``) will raise a ``TypeError``
whereas ``dic.update(arg)`` will silently drop invalid items in ``arg``.

The module also provides a few concrete subclasses that can be useful either:
    - on their own accord
    - as base classes for other more useful subclasses
    - as guides on how to subclass the ABC to create your own dictionary types

Note: the module also defines a custom metaclass that derives from
``ABCMeta``. This should be transparent to client code that does not dig too
deeply into the module's inner workings. For example, potential issues might
arise due to metaclass conflicts, eg in cases where subclassing modules define
their own metaclass or in multiple inheritance scenarios.
"""

__all__ = 'FilterDict', 'StrDict', 'NsDict', 'NestedNsDict'

from functools import update_wrapper, lru_cache
from itertools import chain, starmap, repeat
from collections import defaultdict, OrderedDict as Ord
from collections.abc import *
from abc import ABCMeta, abstractmethod
import logging
from inspect import signature
from types import MappingProxyType
from copy import deepcopy
import re
from weakref import WeakKeyDictionary as Wkd


logger = logging.getLogger(__name__)
__docformat__ = 'restructuredtext'
__version__ = 1, 0, 0


def _iskeyword(k):
    """
    Determines whether a string is a Python keyword.

    :param k: The string to be tested.
    """
    return k in frozenset((
        'False',
        'None',
        'True',
        'and',
        'as',
        'assert',
        'async',
        'await',
        'break',
        'class',
        'continue',
        'def',
        'del',
        'elif',
        'else',
        'except',
        'finally',
        'for',
        'from',
        'global',
        'if',
        'import',
        'in',
        'is',
        'lambda',
        'nonlocal',
        'not',
        'or',
        'pass',
        'raise',
        'return',
        'try',
        'while',
        'with',
        'yield'
    ))

def _getsignature(routine, *implementors, default=None):
    """
    Retrieves the signature of a callable (method/function/etc). Utility
    function that slightly extends the ``signature`` method of the ``inspect``
    module and tries some alternatives when the latter fails (which it does
    with certain ``dict`` methods, for example). Apart from the signature, the
    built-in/stdlib class that contained an implementation of ``routine`` and,
    when possible, the docstring are also retrieved.

    :param routine: Callable whose signature is to be determined.

    :param implementors: Tuple of classes that contain methods of the same
        name as the callable. These serve as a fallback in case the callable
        does not contain enough metadata to accurately retrieve its signature.

    :param default: Default signature, which is ``'self', '*args', '**kw``,
        unless ``routine`` is a ``classmethod`` or ``staticmethod``, in which
        case the first element ``'self'`` is omitted.

        tuple of strings containing the names (and, when
        present, the default values) of all arguments.

    :return: A 3-tuple. First element is the signature, which is itself
        a tuple of strings -- also see ``default`` parameter). Second  the
        implementation (if additional ones are supplied, otherwise ``None``)
        from which the signature was retrieved; and ``routine``'s docstring
        (when available).
    """
    if not callable(routine):
        raise TypeError('First argument must be a callable.')
    name, sig = routine.__name__, None

    if hasattr(routine, '__objclass__'):
        implementors = routine.__objclass__, *implementors

    for c in implementors:
        try:
            f = getattr(c, name)
            doc = f.__doc__
            if not doc:
                raise AttributeError
            if sig:
                return sig, c, doc
            sigobj = signature(f)
            sig = sigobj.parameters.values()
            sig = *map(lambda s: re.sub(r'<.+?>', 'None', str(s)), sig),
            return sig, c, doc
        except (AttributeError, ValueError):
            pass

    if default is None:
        default = (*{
            classmethod: ('cls',),
            staticmethod: ()
        }.get(type(routine), ('self',)), '*args', '**kwargs')
    return default, None, None


class FilterDictMeta(ABCMeta):
    """
    Metaclass for the FilterDict ABC.
    """

    # ***** CONSTANTS: START *****

    # Implementations for these methods are copies of the respective dict ones
    _dic_meths = (

        '__getitem__',
        '__contains__',
        '__iter__',
        '__len__',
        '__delitem__',
        '__reversed__',
        'get',
        'fromkeys',
        'clear',
        'pop',
        'popitem',
        'keys',
        'values',
        'items',
    )

    # These are different to their dict counterparts
    _custom_meths = '__hash__', '__eq__', '__copy__', 'copy', '__deepcopy__'

    # Some properties that one of the IPython helpers inadvertently injects
    # into our namespace in some cases when the module is run or when in an
    # interactive session. These properties are automatically removed.
    _ipython_junk_meths = (
        '_ipython_canary_method_should_not_exist_',
        '_repr_mimebundle_'
    )

    # These SHOULD NOT BE OVERRIDDEN unless you really know what you're
    # doing. Overriding them may result in violation of the "subset" guarantee.
    # guarantee.
    _final_meths = (
        '__setitem__',
        'setdefault',
        'update'
    )

    _abstract_meths = 'keycheck', 'counter_eg'
    _mappingproxy_prop = 'proxy'

    # ***** CONSTANTS: END *****

    _base_name = __all__[0]
    _reserved = frozenset({
        _mappingproxy_prop,
        '__setattr__', '__getattr__', '__getattribute__'
        }.union(
            _dic_meths,
            _ipython_junk_meths,
            _final_meths,
            _custom_meths
        )
    )
    _dics = Wkd()
    _token = object()
    _depth = 0

    def __new__(mcs, clsname, bases, ns):
        base, dics = bases[0], mcs._dics
        base_ns = vars(base)
        kc_s, cnt_eg_s = mcs._abstract_meths

        if base is MutableMapping:
            #  We're at the ABC
            if clsname != mcs._base_name:
                raise ValueError(
                    f'Something went wrong: {clsname} is not the expected ABC.'
                )
            f_base_kc = lambda k: True
            tmp_ns = dict(dics=dics)

            # copy implementations of most methods from respective dict methods
            for m in mcs._dic_meths:
                tmp_ns[m] = dic_bound_meth = f'dict.{m}(dics[self], '
                sig, source_cls, doc = _getsignature(
                    vars(dict)[m], Ord, MutableMapping, Mapping, defaultdict
                )
                selfsig = ', '.join(sig)
                dicparams = ', '.join(
                    map(lambda s: s.partition('=')[0], sig[1:])
                )
                dicsig = f'{dic_bound_meth}{dicparams})'
                exec(
                    f'def {m}({selfsig}):\n'
                    f'    return {dicsig}\n',
                    tmp_ns
                )
                ns[m] = f = tmp_ns[m]

                # setting appropriate metadata
                f.__doc__ = doc
                f.__module__ = mcs.__module__
                f.__qualname__ = f'{clsname}.{m}'
        else:
            # see which of the "final" methods have been overridden
            overriden = set(mcs._final_meths) & set(ns)
            if overriden:
                logger.warning(
                    f'Dictionary setter methods {", ".join(overriden)} should '
                    f'not be be overriden. Doing so may break the constraint '
                    f'guarantees of the inheritance tree.'
                )
            f_base_kc = base_ns[kc_s]
            if not callable(f_base_kc):
                f_base_kc = f_base_kc.__func__

        if kc_s in ns:
            f_kc = ns[kc_s].__func__
            cnt_eg_val = ns[cnt_eg_s].__func__()
            tmp = f'.{kc_s}({cnt_eg_s}) must be'
            if not f_base_kc(cnt_eg_val):
                raise ValueError(f'{base}{tmp} True')
            if f_kc(cnt_eg_val):
                raise ValueError(f'{clsname}{tmp} False')

            # Conjunction of cls.keycheck(k) with basecls.keycheck(k) (after
            # checking k against a list of reserved names). This allows us to
            # specify only the tests that differentiate cls from basecls. This
            # avoids repetition of code (cls.keycheck would otherwise always
            # have to call super().keycheck explictly before the actual,
            # subclass-specific checks.
            def _total_kc(k):
                return (
                    k not in mcs._reserved and f_base_kc(k) and f_kc(k)
                )
            ns[kc_s] = staticmethod(update_wrapper(_total_kc, f_kc))

        else:
            ns[kc_s] = f_base_kc
            ns[cnt_eg_s] = base_ns[cnt_eg_s]

        ns[mcs._mappingproxy_prop] = property(
            fget=lambda self: MappingProxyType(dics[self]),
            doc=f'A read-only proxy of the underlying mapping (an instance '
                f'of OrderedDict) used to implement {clsname}. May be useful '
                f'for iterating over large mappings as it bypasses the custom '
                f'__getitem__ defined in the ABC and used for all item access.'
        )

        return super().__new__(mcs, clsname, bases, ns)


# Save typing and make refactoring easier by storing the metaclass in a local
_m = FilterDictMeta
_d = _m._dics
_tok = _m._token


class FilterDict(MutableMapping, metaclass=_m):
    """
    Abstract Base Class (ABC) that represents a mutable (ie non-read-only)
    mapping (aka dictionary) some utility class/static methods and, more
    importantly, greater flexibility in constructor parameter specification
    (see the documentation for ``update``).
    """
    @staticmethod
    @abstractmethod
    def keycheck(k):
        """
        Checks whether ``k`` is a valid key for the dictionary. This is an
        abstract method that nevertheless provides a default implementation
        in which any ``k`` that is a valid dictionary key (hashable, immutable)
        passes the test. Subclasses are *required* to override this method to
        implement their own (more stringent) checks and thus reduce the set of
        acceptable keys. Also see ``counter_eg``.

        :param k: The candidate key to validate.

        :return: In this ABC implementation, the method returns True if ``k``
            is hashable.
        """
        try:
            # this should always be True, as it is basically ``not set()``...
            return not {k}.clear()
        # ...unless there is an error (eg unhashable k)
        except (KeyError,TypeError):
            return False

    @staticmethod
    @abstractmethod
    def counter_eg():
        """
        Provides a "counter-example", ie an object that is considered a valid
        key for the parent class but is invalid for the concrete subclass.
        :return:
        """
        return set()

    def __new__(cls, *args, **kwargs):
        inst = super().__new__(cls)
        _d[inst] = {}
        # If you're tempted to change this to ``inst.update(*args, **kwargs)``,
        # don't. Doing so will most likely break custom implementations of
        # __getattribute__ and/or __getattr__ in subclasses.
        __class__.update(inst, *args, **kwargs)
        return inst

    def __str__(self):
        kwa = ', \n\t'.join(starmap(
            lambda k, v: f'{k!r}: {v!r}',
            self.items()
        ))
        if kwa:
            kwa = f'\n\t{kwa}\n'
        return f'{type(self).__name__}({{{kwa}}})'

    def update(self, *args, **kwargs):
        """
        Updates dictionary with elements from sequences or mappings. Similar
        to ``dict.update`` but with an extended signature that allows ``args`` to be a mixed sequence of sequences or mappings. In
        other words, each elements in ``args`` is a valid input for
        ``dict.update``.

        :param args: Sequence of ``Mapping``s, ``Sequence``s of key-value
            pairs or ``ItemView``s, in any order. Each element ``e`` in
            ``args`` is valid if it is accepted by dict.update.

        :param kwargs: Keywords are validated with ``keycheck``; those items
            that fail the test are silently excluded from the dictionary.
        """
        c = type(self)
        kc, tmp = c.keycheck, Ord()
        tmp_upd, tmp_popitem = tmp.update, tmp.popitem,
        for a in args:
            try:
                tmp_upd(a)
            except (TypeError, ValueError):
                pass
        tmp_upd(kwargs)
        try:
            # OrderedDict.popitem called with the optional argument False
            # allows for the dictionary to be constructed as the temporary
            # mapping is destroyed. This keeps memory consumption at a minimum.
            _d[self].update(
                (k, v) for k, v in map(tmp_popitem, repeat(False)) if kc(k)
            )
        except KeyError:
            pass

    def __setitem__(self, k, v):
        c = type(self)
        if not c.keycheck(k):
            raise TypeError(f"Invalid key '{k}'")
        _d[self][k] = v

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return (
            self is other or
            isinstance(other, Mapping) and self.items() == other.items()
        )

    __copy__ = copy = lambda self: type(self)(self)

    def __deepcopy__(self, memo=None):
        return type(self)(deepcopy(_d[self], memo))


class StrDict(FilterDict):
    """
    Dictionary that only accepts strings as keys.
    """
    @staticmethod
    def keycheck(k):
        """
        Checks whether ``k`` is a valid key for the dictionary. In this class,
        any string is considered a valid key, and anything else an invalid one.
        """
        return isinstance(k, str)

    @staticmethod
    def counter_eg():
        """

        :return: ``None`` (just a value
        """
        return


class NsDict(StrDict):
    """
    Namespace-like dictionary, similar to ``SimpleNamespace`` from the
    ``types`` module.
    """
    @staticmethod
    def keycheck(k):
        """
        Determines whether an object is a valid identifier, meaning that it's
        a string that starts with a letter or underscore and is followed by
        zero or more letters, numbers or underscores. The requirement for
        being a string is implicit by its inheriting from StrDict and does not
        need to be encoded (by something like isinstance(k, str)) here.

        :param k: The key to be checked

        :return: Boolean result
        """
        return k.isidentifier() and not _iskeyword(k)

    @staticmethod
    def counter_eg():
        """

        :return: The empty string, which is a valid key for StrDict but not
        a valid attribute name (and thus key in a namespace dictionary, such
        as obj.__dict__)
        """
        return ''

    def __str__(self):
        cls = type(self)
        try:
            _m._depth += 1
            ind_l = '\t' * _m._depth
            ind_s = ind_l[:-1]
            kwa = f', \n{ind_l}'.join(starmap(
                lambda k, v: f'{k}={v!r}',
                _d[self].items()
            ))
            if kwa:
                kwa = f'\n{ind_l}{kwa}\n{ind_s}'
            _m._depth -= 1
            return f'{cls.__name__}({kwa})'
        finally:
            _m._depth = 0

    def __repr__(self):
        return re.sub(r'[\t\n]+', '', str(self))

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        try:
            self[k] = v
        except KeyError:
            raise AttributeError(k)

    def __delattr__(self, k):
        try:
            del self[k]
        except KeyError:
            raise AttributeError(k)


class NestedNsDict(NsDict):
    """
    Namespace-like dictionary that also supports chained assignment, eg
    >>> nns = NestedNsDict()
    >>> nns.a.b.c = 42
    >>> nns.a.b.c
    42

    Parent dictionary can be retrieved via the ``parent`` attribute. If there
    is no parent, ``None`` is returned:
    >>> nns.a.b.parent == nns.a
    True

    >>> nns.parent is None
    True

    """
    def __getattr__(self, k):
        parents = __class__._parents
        if k == 'parent':
            return parents.get(self)
        try:
            return self[k]
        except KeyError:
            # if k is missing, create a new instance. As before, we use
            # type(self)(), not __class__(), as the latter would not work
            # correctly if self is an instance of a subclass of NestedNsDict
            self[k] = d = type(self)()
            parents[d] = self
            return d

    def __setattr__(self, k, v):
        if k == 'parent':
            raise ValueError(f"cannot use reserved name '{k}'")
        super().__setattr__(k, v)

    def __init__(self, *a, **kwa):
        super().__init__(*a, **kwa)
        c = __class__
        if not hasattr(c, '_parents'):
            c._parents = Wkd()
        c._parents[self] = None
