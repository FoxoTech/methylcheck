"""
Multi-dimensional Scaling (MDS)
Copied from scikit-learn v0.22.1 without changes (just all of the functions in one place for MDS to work)
"""

# author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# License: BSD
__version__ = '0.22.1'

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
import warnings
import copy
from collections import defaultdict
import inspect
import re
import os
import numbers
import pprint

# validation
import scipy.sparse as sp

_global_config = {
    'assume_finite': bool(os.environ.get('SKLEARN_ASSUME_FINITE', False)),
    'working_memory': int(os.environ.get('SKLEARN_WORKING_MEMORY', 1024)),
    'print_changed_only': False,
}

def get_config():
    """Retrieve current values for configuration set by :func:`set_config`

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context: Context manager for global scikit-learn configuration
    set_config: Set global scikit-learn configuration
    """
    return _global_config.copy()


class BaseEstimator:
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn('From version 0.24, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute. Previously '
                              'it would return None.',
                              FutureWarning)
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self, N_CHAR_MAX=700):
        # N_CHAR_MAX is the (approximate) maximum number of non-blank
        # characters to render. We pass it as an optional parameter to ease
        # the tests.

        #from .utils._pprint import _EstimatorPrettyPrinter

        N_MAX_ELEMENTS_TO_SHOW = 30  # number of elements to show in sequences

        # use ellipsis for sequences with a lot of elements
        pp = _EstimatorPrettyPrinter(
            compact=True, indent=1, indent_at_name=True,
            n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW)

        repr_ = pp.pformat(self)

        # Use bruteforce ellipsis when there are a lot of non-blank characters
        n_nonblank = len(''.join(repr_.split()))
        if n_nonblank > N_CHAR_MAX:
            lim = N_CHAR_MAX // 2  # apprx number of chars to keep on both ends
            regex = r'^(\s*\S){%d}' % lim
            # The regex '^(\s*\S){%d}' % n
            # matches from the start of the string until the nth non-blank
            # character:
            # - ^ matches the start of string
            # - (pattern){n} matches n repetitions of pattern
            # - \s*\S matches a non-blank char following zero or more blanks
            left_lim = re.match(regex, repr_).end()
            right_lim = re.match(regex, repr_[::-1]).end()

            if '\n' in repr_[left_lim:-right_lim]:
                # The left side and right side aren't on the same line.
                # To avoid weird cuts, e.g.:
                # categoric...ore',
                # we need to start the right side with an appropriate newline
                # character so that it renders properly as:
                # categoric...
                # handle_unknown='ignore',
                # so we add [^\n]*\n which matches until the next \n
                regex += r'[^\n]*\n'
                right_lim = re.match(regex, repr_[::-1]).end()

            ellipsis = '...'
            if left_lim + len(ellipsis) < len(repr_) - right_lim:
                # Only add ellipsis if it results in a shorter repr
                repr_ = repr_[:left_lim] + '...' + repr_[-right_lim:]

        return repr_

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if type(self).__module__.startswith('sklearn.'):
            return dict(state.items(), _sklearn_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        if type(self).__module__.startswith('sklearn.'):
            pickle_version = state.pop("_sklearn_version", "pre-0.18")
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning)
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

    def _more_tags(self):
        return _DEFAULT_TAGS

    def _get_tags(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags




class KeyValTuple(tuple):
    """Dummy class for correctly rendering key-value tuples from dicts."""
    def __repr__(self):
        # needed for _dispatch[tuple.__repr__] not to be overridden
        return super().__repr__()

class KeyValTupleParam(KeyValTuple):
    """Dummy class for correctly rendering key-value tuples from parameters."""
    pass

def _safe_repr(object, context, maxlevels, level, changed_only=False):
    """Same as the builtin _safe_repr, with added support for Estimator
    objects."""
    typ = type(object)

    if typ in pprint._builtin_scalars:
        return repr(object), True, False

    r = getattr(typ, "__repr__", None)
    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return "{}", True, False
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(object.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only)
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only)
            append("%s: %s" % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return "{%s}" % ", ".join(components), readable, recursive

    if (issubclass(typ, list) and r is list.__repr__) or \
       (issubclass(typ, tuple) and r is tuple.__repr__):
        if issubclass(typ, list):
            if not object:
                return "[]", True, False
            format = "[%s]"
        elif len(object) == 1:
            format = "(%s,)"
        else:
            if not object:
                return "()", True, False
            format = "(%s)"
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return format % "...", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            orepr, oreadable, orecur = _safe_repr(
                o, context, maxlevels, level, changed_only=changed_only)
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return format % ", ".join(components), readable, recursive

    if issubclass(typ, BaseEstimator):
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        if changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(params.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only)
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only)
            append("%s=%s" % (krepr.strip("'"), vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ("%s(%s)" % (typ.__name__, ", ".join(components)), readable,
                recursive)

    rep = repr(object)
    return rep, (rep and not rep.startswith('<')), False

import pprint

class _EstimatorPrettyPrinter(pprint.PrettyPrinter):
    """Pretty Printer class for estimator objects.
    This extends the pprint.PrettyPrinter class, because:
    - we need estimators to be printed with their parameters, e.g.
      Estimator(param1=value1, ...) which is not supported by default.
    - the 'compact' parameter of PrettyPrinter is ignored for dicts, which
      may lead to very long representations that we want to avoid.
    Quick overview of pprint.PrettyPrinter (see also
    https://stackoverflow.com/questions/49565047/pprint-with-hex-numbers):
    - the entry point is the _format() method which calls format() (overridden
      here)
    - format() directly calls _safe_repr() for a first try at rendering the
      object
    - _safe_repr formats the whole object reccursively, only calling itself,
      not caring about line length or anything
    - back to _format(), if the output string is too long, _format() then calls
      the appropriate _pprint_TYPE() method (e.g. _pprint_list()) depending on
      the type of the object. This where the line length and the compact
      parameters are taken into account.
    - those _pprint_TYPE() methods will internally use the format() method for
      rendering the nested objects of an object (e.g. the elements of a list)
    In the end, everything has to be implemented twice: in _safe_repr and in
    the custom _pprint_TYPE methods. Unfortunately PrettyPrinter is really not
    straightforward to extend (especially when we want a compact output), so
    the code is a bit convoluted.
    This class overrides:
    - format() to support the changed_only parameter
    - _safe_repr to support printing of estimators (for when they fit on a
      single line)
    - _format_dict_items so that dict are correctly 'compacted'
    - _format_items so that ellipsis is used on long lists and tuples
    When estimators cannot be printed on a single line, the builtin _format()
    will call _pprint_estimator() because it was registered to do so (see
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator).
    both _format_dict_items() and _pprint_estimator() use the
    _format_params_or_dict_items() method that will format parameters and
    key-value pairs respecting the compact parameter. This method needs another
    subroutine _pprint_key_val_tuple() used when a parameter or a key-value
    pair is too long to fit on a single line. This subroutine is called in
    _format() and is registered as well in the _dispatch dict (just like
    _pprint_estimator). We had to create the two classes KeyValTuple and
    KeyValTupleParam for this.
    """

    def __init__(self, indent=1, width=80, depth=None, stream=None, *,
                 compact=False, indent_at_name=True,
                 n_max_elements_to_show=None):
        super().__init__(indent, width, depth, stream, compact=compact)
        self._indent_at_name = indent_at_name
        if self._indent_at_name:
            self._indent_per_level = 1  # ignore indent param
        self._changed_only = get_config()['print_changed_only']
        # Max number of elements in a list, dict, tuple until we start using
        # ellipsis. This also affects the number of arguments of an estimators
        # (they are treated as dicts)
        self.n_max_elements_to_show = n_max_elements_to_show

    def format(self, object, context, maxlevels, level):
        return _safe_repr(object, context, maxlevels, level,
                          changed_only=self._changed_only)

    def _pprint_estimator(self, object, stream, indent, allowance, context,
                          level):
        stream.write(object.__class__.__name__ + '(')
        if self._indent_at_name:
            indent += len(object.__class__.__name__)

        if self._changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)

        params = OrderedDict((name, val)
                             for (name, val) in sorted(params.items()))

        self._format_params(params.items(), stream, indent, allowance + 1,
                            context, level)
        stream.write(')')

    def _format_dict_items(self, items, stream, indent, allowance, context,
                           level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=True)

    def _format_params(self, items, stream, indent, allowance, context, level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=False)

    def _format_params_or_dict_items(self, object, stream, indent, allowance,
                                     context, level, is_dict):
        """Format dict items or parameters respecting the compact=True
        parameter. For some reason, the builtin rendering of dict items doesn't
        respect compact=True and will use one line per key-value if all cannot
        fit in a single line.
        Dict items will be rendered as <'key': value> while params will be
        rendered as <key=value>. The implementation is mostly copy/pasting from
        the builtin _format_items().
        This also adds ellipsis if the number of items is greater than
        self.n_max_elements_to_show.
        """
        write = stream.write
        indent += self._indent_per_level
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(object)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(', ...')
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                k, v = ent
                krepr = self._repr(k, context, level)
                vrepr = self._repr(v, context, level)
                if not is_dict:
                    krepr = krepr.strip("'")
                middle = ': ' if is_dict else '='
                rep = krepr + middle + vrepr
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            class_ = KeyValTuple if is_dict else KeyValTupleParam
            self._format(class_(ent), stream, indent,
                         allowance if last else 1, context, level)

    def _format_items(self, items, stream, indent, allowance, context, level):
        """Format the items of an iterable (list, tuple...). Same as the
        built-in _format_items, with support for ellipsis if the number of
        elements is greater than self.n_max_elements_to_show.
        """
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(', ...')
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent,
                         allowance if last else 1, context, level)

    def _pprint_key_val_tuple(self, object, stream, indent, allowance, context,
                              level):
        """Pretty printing for key-value tuples from dict or parameters."""
        k, v = object
        rep = self._repr(k, context, level)
        if isinstance(object, KeyValTupleParam):
            rep = rep.strip("'")
            middle = '='
        else:
            middle = ': '
        stream.write(rep)
        stream.write(middle)
        self._format(v, stream, indent + len(rep) + len(middle), allowance,
                     context, level)

    # Note: need to copy _dispatch to prevent instances of the builtin
    # PrettyPrinter class to call methods of _EstimatorPrettyPrinter (see issue
    # 12906)
    # mypy error: "Type[PrettyPrinter]" has no attribute "_dispatch"
    _dispatch = pprint.PrettyPrinter._dispatch.copy()  # type: ignore
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator
    _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple




#from ..metrics import euclidean_distances
# Utility Functions
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not sp.issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not sp.issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype

def check_pairwise_arrays(X, Y, precomputed=False, dtype=None,
                          accept_sparse='csr', force_all_finite=True,
                          copy=False):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_a, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)

    precomputed : bool
        True if X is to be treated as precomputed distances to the samples in
        Y.

    dtype : string, type, list of types or None (default=None)
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

        .. versionadded:: 0.18

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

        .. versionadded:: 0.22
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    copy : bool
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

        .. versionadded:: 0.22

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    X, Y, dtype_float = _return_float_dtype(X, Y)

    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                            copy=copy, force_all_finite=force_all_finite,
                            estimator=estimator)
    else:
        X = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                        copy=copy, force_all_finite=force_all_finite,
                        estimator=estimator)
        Y = check_array(Y, accept_sparse=accept_sparse, dtype=dtype,
                        copy=copy, force_all_finite=force_all_finite,
                        estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.

    Performs no input validation.

    Parameters
    ----------
    X : array_like
        The input array
    squared : bool, optional (default = False)
        If True, return squared norms.

    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """
    if sp.issparse(X):
        if not isinstance(X, sp.csr_matrix):
            X = sp.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms

def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, (default=False)
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sp.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sp.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (sp.issparse(a) and sp.issparse(b)
            and dense_output and hasattr(ret, "toarray")):
        return ret.toarray()
    return ret

# Pairwise distances
def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
                        X_norm_squared=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.

    However, this is not the most precise way of doing this computation, and
    the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

    Y_norm_squared : array-like, shape (n_samples_2, ), optional
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    squared : boolean, optional
        Return squared Euclidean distances.

    X_norm_squared : array-like of shape (n_samples,), optional
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    Notes
    -----
    To achieve better accuracy, `X_norm_squared` and `Y_norm_squared` may be
    unused if they are passed as ``float32``.

    Returns
    -------
    distances : array, shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])

    See also
    --------
    paired_distances : distances betweens pairs of elements of X and Y.
    """
    X, Y = check_pairwise_arrays(X, Y)

    # If norms are passed as float32, they are unused. If arrays are passed as
    # float32, norms needs to be recomputed on upcast chunks.
    # TODO: use a float64 accumulator in row_norms to avoid the latter.
    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
        if XX.dtype == np.float32:
            XX = None
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if X is Y and XX is not None:
        # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
        if YY.dtype == np.float32:
            YY = None
    elif Y.dtype == np.float32:
        YY = None
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    if X.dtype == np.float32:
        # To minimize precision issues with float32, we compute the distance
        # matrix on chunks of X and Y upcast to float64
        distances = _euclidean_distances_upcast(X, XX, Y, YY)
    else:
        # if dtype is already float64, no need to chunk and upcast
        distances = - 2 * safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)

#from ..utils import check_random_state, check_array, check_symmetric
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def _assert_all_finite(X, allow_nan=False, msg_dtype=None):
    """Like assert_all_finite, but only for ndarray."""

    # validation is also imported in extmath
    #from .extmath import _safe_accumulator_op
    # Use at least float64 for the accumulating functions to avoid precision issue
    # see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
    # as it is in case the float overflows
    def _safe_accumulator_op(op, x, *args, **kwargs):
        """
        This function provides numpy accumulator functions with a float64 dtype
        when used on a floating point input. This prevents accumulator overflow on
        smaller floating point dtypes.

        Parameters
        ----------
        op : function
            A numpy accumulator function such as np.mean or np.sum
        x : numpy array
            A numpy array to apply the accumulator function
        *args : positional arguments
            Positional arguments passed to the accumulator function after the
            input x
        **kwargs : keyword arguments
            Keyword arguments passed to the accumulator function

        Returns
        -------
        result : The output of the accumulator function passed to this function
        """
        if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
            result = op(x, *args, **kwargs, dtype=np.float64)
        else:
            result = op(x, *args, **kwargs)
        return result

    if get_config()['assume_finite']:
        return
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    is_float = X.dtype.kind in 'fc'
    if is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(
                    msg_err.format
                    (type_err,
                     msg_dtype if msg_dtype is not None else X.dtype)
            )
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype('object') and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")

def assert_all_finite(X, allow_nan=False):
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix

    allow_nan : bool
    """
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)

def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)

def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=None, estimator=None):

    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

        For object dtyped data, only np.nan is checked and not np.inf.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if array is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow array.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean or None, optional (default=None)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

        .. deprecated:: 0.21
            ``warn_on_dtype`` is deprecated in version 0.21 and will be
            removed in 0.23.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    # warn_on_dtype deprecation
    if warn_on_dtype is not None:
        warnings.warn(
            "'warn_on_dtype' is deprecated in version 0.21 and will be "
            "removed in 0.23. Don't set `warn_on_dtype` to remove this "
            "warning.",
            FutureWarning, stacklevel=2)

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
        dtypes_orig = list(array.dtypes)
        # pandas boolean dtype __array__ interface coerces bools to objects
        for i, dtype_iter in enumerate(dtypes_orig):
            if dtype_iter.kind == 'b':
                dtypes_orig[i] = np.object

        if all(isinstance(dtype, np.dtype) for dtype in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', np.ComplexWarning)
                if dtype is not None and np.dtype(dtype).kind in 'iu':
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = np.asarray(array, order=order)
                    if array.dtype.kind == 'f':
                        _assert_all_finite(array, allow_nan=False,
                                           msg_dtype=dtype)
                    array = array.astype(dtype, casting="unsafe", copy=False)
                else:
                    array = np.asarray(array, order=order, dtype=dtype)
            except np.ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning, stacklevel=2)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))

        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning, stacklevel=2)

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        # if there was at the beginning some other types than the final one
        # (for instance in a DataFrame that can contain several dtypes) then
        # some data must have been converted
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        warnings.warn(msg, DataConversionWarning, stacklevel=3)

    return array

def check_symmetric(array, tol=1E-10, raise_warning=True,
                    raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : nd-array or sparse matrix
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : boolean (default=True)
        If True then raise a warning if conversion is required.
    raise_exception : boolean (default=False)
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : ndarray or sparse matrix
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError("array must be 2-dimensional and square. "
                         "shape = {0}".format(array.shape))

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ['csr', 'csc', 'coo']:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn("Array is not symmetric, and will be converted "
                          "to symmetric by average with its transpose.",
                          stacklevel=2)
        if sp.issparse(array):
            conversion = 'to' + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array

#from ..isotonic import IsotonicRegression
def isotonic_regression(y, sample_weight=None, y_min=None, y_max=None,
                        increasing=True):
    """Solve the isotonic regression model::

        min sum w[i] (y[i] - y_[i]) ** 2

        subject to y_min = y_[1] <= y_[2] ... <= y_[n] = y_max

    where:
        - y[i] are inputs (real numbers)
        - y_[i] are fitted
        - w[i] are optional strictly positive weights (default to 1.0)

    Read more in the :ref:`User Guide <isotonic>`.

    Parameters
    ----------
    y : iterable of floats
        The data.

    sample_weight : iterable of floats, optional, default: None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).

    y_min : optional, default: None
        If not None, set the lowest value of the fit to y_min.

    y_max : optional, default: None
        If not None, set the highest value of the fit to y_max.

    increasing : boolean, optional, default: True
        Whether to compute ``y_`` is increasing (if set to True) or decreasing
        (if set to False)

    Returns
    -------
    y_ : list of floats
        Isotonic fit of y.

    References
    ----------
    "Active set algorithms for isotonic regression; A unifying framework"
    by Michael J. Best and Nilotpal Chakravarti, section 3.
    """
    order = np.s_[:] if increasing else np.s_[::-1]
    y = check_array(y, ensure_2d=False, dtype=[np.float64, np.float32])
    y = np.array(y[order], dtype=y.dtype)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=y.dtype)
    else:
        sample_weight = np.array(sample_weight[order], dtype=y.dtype)

    _inplace_contiguous_isotonic_regression(y, sample_weight)
    if y_min is not None or y_max is not None:
        # Older versions of np.clip don't accept None as a bound, so use np.inf
        if y_min is None:
            y_min = -np.inf
        if y_max is None:
            y_max = np.inf
        np.clip(y, y_min, y_max, y)
    return y[order]

class RegressorMixin:
    """Mixin class for all regression estimators in scikit-learn."""
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix or a list of generic objects instead,
            shape = (n_samples, n_samples_fitted),
            where n_samples_fitted is the number of
            samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.

        Notes
        -----
        The R2 score used when calling ``score`` on a regressor will use
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with :func:`~sklearn.metrics.r2_score`. This will influence the
        ``score`` method of all the multioutput regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`). To specify the
        default value manually and avoid the warning, please either call
        :func:`~sklearn.metrics.r2_score` directly or make a custom scorer with
        :func:`~sklearn.metrics.make_scorer` (the built-in scorer ``'r2'`` uses
        ``multioutput='uniform_average'``).
        """

        from .metrics import r2_score
        from .metrics._regression import _check_reg_targets
        y_pred = self.predict(X)
        # XXX: Remove the check in 0.23
        y_type, _, _, _ = _check_reg_targets(y, y_pred, None)
        if y_type == 'continuous-multioutput':
            warnings.warn("The default value of multioutput (not exposed in "
                          "score method) will change from 'variance_weighted' "
                          "to 'uniform_average' in 0.23 to keep consistent "
                          "with 'metrics.r2_score'. To specify the default "
                          "value manually and avoid the warning, please "
                          "either call 'metrics.r2_score' directly or make a "
                          "custom scorer with 'metrics.make_scorer' (the "
                          "built-in scorer 'r2' uses "
                          "multioutput='uniform_average').", FutureWarning)
        return r2_score(y, y_pred, sample_weight=sample_weight,
                        multioutput='variance_weighted')

class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    """Isotonic regression model.

    The isotonic regression optimization problem is defined by::

        min sum w_i (y[i] - y_[i]) ** 2

        subject to y_[i] <= y_[j] whenever X[i] <= X[j]
        and min(y_) = y_min, max(y_) = y_max

    where:
        - ``y[i]`` are inputs (real numbers)
        - ``y_[i]`` are fitted
        - ``X`` specifies the order.
          If ``X`` is non-decreasing then ``y_`` is non-decreasing.
        - ``w[i]`` are optional strictly positive weights (default to 1.0)

    Read more in the :ref:`User Guide <isotonic>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    y_min : optional, default: None
        If not None, set the lowest value of the fit to y_min.

    y_max : optional, default: None
        If not None, set the highest value of the fit to y_max.

    increasing : boolean or string, optional, default: True
        If boolean, whether or not to fit the isotonic regression with y
        increasing or decreasing.

        The string value "auto" determines whether y should
        increase or decrease based on the Spearman correlation estimate's
        sign.

    out_of_bounds : string, optional, default: "nan"
        The ``out_of_bounds`` parameter handles how x-values outside of the
        training domain are handled.  When set to "nan", predicted y-values
        will be NaN.  When set to "clip", predicted y-values will be
        set to the value corresponding to the nearest train interval endpoint.
        When set to "raise", allow ``interp1d`` to throw ValueError.


    Attributes
    ----------
    X_min_ : float
        Minimum value of input array `X_` for left bound.

    X_max_ : float
        Maximum value of input array `X_` for right bound.

    f_ : function
        The stepwise interpolating function that covers the input domain ``X``.

    Notes
    -----
    Ties are broken using the secondary method from Leeuw, 1977.

    References
    ----------
    Isotonic Median Regression: A Linear Programming Approach
    Nilotpal Chakravarti
    Mathematics of Operations Research
    Vol. 14, No. 2 (May, 1989), pp. 303-308

    Isotone Optimization in R : Pool-Adjacent-Violators
    Algorithm (PAVA) and Active Set Methods
    Leeuw, Hornik, Mair
    Journal of Statistical Software 2009

    Correctness of Kruskal's algorithms for monotone regression with ties
    Leeuw, Psychometrica, 1977

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.isotonic import IsotonicRegression
    >>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
    >>> iso_reg = IsotonicRegression().fit(X.flatten(), y)
    >>> iso_reg.predict([.1, .2])
    array([1.8628..., 3.7256...])
    """
    def __init__(self, y_min=None, y_max=None, increasing=True,
                 out_of_bounds='nan'):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def _check_fit_data(self, X, y, sample_weight=None):
        if len(X.shape) != 1:
            raise ValueError("X should be a 1d array")

    def _build_f(self, X, y):
        """Build the f_ interp1d function."""

        # Handle the out_of_bounds argument by setting bounds_error
        if self.out_of_bounds not in ["raise", "nan", "clip"]:
            raise ValueError("The argument ``out_of_bounds`` must be in "
                             "'nan', 'clip', 'raise'; got {0}"
                             .format(self.out_of_bounds))

        bounds_error = self.out_of_bounds == "raise"
        if len(y) == 1:
            # single y, constant prediction
            self.f_ = lambda x: y.repeat(x.shape)
        else:
            self.f_ = interpolate.interp1d(X, y, kind='linear',
                                           bounds_error=bounds_error)

    def _build_y(self, X, y, sample_weight, trim_duplicates=True):
        """Build the y_ IsotonicRegression."""
        self._check_fit_data(X, y, sample_weight)

        # Determine increasing if auto-determination requested
        if self.increasing == 'auto':
            self.increasing_ = check_increasing(X, y)
        else:
            self.increasing_ = self.increasing

        # If sample_weights is passed, removed zero-weight values and clean
        # order
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False,
                                        dtype=X.dtype)
            mask = sample_weight > 0
            X, y, sample_weight = X[mask], y[mask], sample_weight[mask]
        else:
            sample_weight = np.ones(len(y), dtype=X.dtype)

        order = np.lexsort((y, X))
        X, y, sample_weight = [array[order] for array in [X, y, sample_weight]]
        unique_X, unique_y, unique_sample_weight = _make_unique(
            X, y, sample_weight)

        # Store _X_ and _y_ to maintain backward compat during the deprecation
        # period of X_ and y_
        self._X_ = X = unique_X
        self._y_ = y = isotonic_regression(unique_y, unique_sample_weight,
                                           self.y_min, self.y_max,
                                           increasing=self.increasing_)

        # Handle the left and right bounds on X
        self.X_min_, self.X_max_ = np.min(X), np.max(X)

        if trim_duplicates:
            # Remove unnecessary points for faster prediction
            keep_data = np.ones((len(y),), dtype=bool)
            # Aside from the 1st and last point, remove points whose y values
            # are equal to both the point before and the point after it.
            keep_data[1:-1] = np.logical_or(
                np.not_equal(y[1:-1], y[:-2]),
                np.not_equal(y[1:-1], y[2:])
            )
            return X[keep_data], y[keep_data]
        else:
            # The ability to turn off trim_duplicates is only used to it make
            # easier to unit test that removing duplicates in y does not have
            # any impact the resulting interpolation function (besides
            # prediction speed).
            return X, y

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as :meth:`transform` needs X to interpolate
        new input data.
        """
        check_params = dict(accept_sparse=False, ensure_2d=False)
        X = check_array(X, dtype=[np.float64, np.float32], **check_params)
        y = check_array(y, dtype=X.dtype, **check_params)
        check_consistent_length(X, y, sample_weight)

        # Transform y by running the isotonic regression algorithm and
        # transform X accordingly.
        X, y = self._build_y(X, y, sample_weight)

        # It is necessary to store the non-redundant part of the training set
        # on the model to make it possible to support model persistence via
        # the pickle module as the object built by scipy.interp1d is not
        # picklable directly.
        self._necessary_X_, self._necessary_y_ = X, y

        # Build the interpolation function
        self._build_f(X, y)
        return self

    def transform(self, T):
        """Transform new data by linear interpolation

        Parameters
        ----------
        T : array-like of shape (n_samples,)
            Data to transform.

        Returns
        -------
        T_ : array, shape=(n_samples,)
            The transformed data
        """

        if hasattr(self, '_necessary_X_'):
            dtype = self._necessary_X_.dtype
        else:
            dtype = np.float64

        T = check_array(T, dtype=dtype, ensure_2d=False)

        if len(T.shape) != 1:
            raise ValueError("Isotonic regression input should be a 1d array")

        # Handle the out_of_bounds argument by clipping if needed
        if self.out_of_bounds not in ["raise", "nan", "clip"]:
            raise ValueError("The argument ``out_of_bounds`` must be in "
                             "'nan', 'clip', 'raise'; got {0}"
                             .format(self.out_of_bounds))

        if self.out_of_bounds == "clip":
            T = np.clip(T, self.X_min_, self.X_max_)

        res = self.f_(T)

        # on scipy 0.17, interp1d up-casts to float64, so we cast back
        res = res.astype(T.dtype)

        return res

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,)
            Data to transform.

        Returns
        -------
        T_ : array, shape=(n_samples,)
            Transformed data.
        """
        return self.transform(T)

    def __getstate__(self):
        """Pickle-protocol - return state of the estimator. """
        state = super().__getstate__()
        # remove interpolation method
        state.pop('f_', None)
        return state

    def __setstate__(self, state):
        """Pickle-protocol - set state of the estimator.

        We need to rebuild the interpolation function.
        """
        super().__setstate__(state)
        if hasattr(self, '_necessary_X_') and hasattr(self, '_necessary_y_'):
            self._build_f(self._necessary_X_, self._necessary_y_)

    def _more_tags(self):
        return {'X_types': ['1darray']}

def _smacof_single(dissimilarities, metric=True, n_components=2, init=None,
                   max_iter=300, verbose=0, eps=1e-3, random_state=None):
    """Computes multidimensional scaling using SMACOF algorithm

    Parameters
    ----------
    dissimilarities : ndarray, shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : boolean, optional, default: True
        Compute metric or nonmetric SMACOF algorithm.

    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray, shape (n_samples, n_components), optional, default: None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, optional, default: 0
        Level of verbosity.

    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    random_state : int, RandomState instance or None, optional, default: None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Returns
    -------
    X : ndarray, shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    n_iter : int
        The number of iterations corresponding to the best stress.
    """
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)

    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()
    sim_flat_w = sim_flat[sim_flat != 0]
    if init is None:
        # Randomly choose initial configuration
        X = random_state.rand(n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" %
                             (n_samples, n_components))
        X = init

    old_stress = None
    ir = IsotonicRegression()
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)

        if metric:
            disparities = dissimilarities
        else:
            dis_flat = dis.ravel()
            # dissimilarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
                                   (disparities ** 2).sum())

        # Compute stress
        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2

        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = - ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1. / n_samples * np.dot(B, X)

        dis = np.sqrt((X ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))
        if old_stress is not None:
            if(old_stress - stress / dis) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break
        old_stress = stress / dis

    return X, stress, it + 1


def smacof(dissimilarities, metric=True, n_components=2, init=None, n_init=8,
           n_jobs=None, max_iter=300, verbose=0, eps=1e-3, random_state=None,
           return_n_iter=False):
    """Computes multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can summarized by the following steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    dissimilarities : ndarray, shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : boolean, optional, default: True
        Compute metric or nonmetric SMACOF algorithm.

    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray, shape (n_samples, n_components), optional, default: None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, optional, default: 8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, optional, default: 0
        Level of verbosity.

    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    random_state : int, RandomState instance or None, optional, default: None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    return_n_iter : bool, optional, default: False
        Whether or not to return the number of iterations.

    Returns
    -------
    X : ndarray, shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.

    Notes
    -----
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)

    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)

    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)

    if hasattr(init, '__array__'):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                'Explicit initial positions passed: '
                'performing only one init of the MDS instead of %d'
                % n_init)
            n_init = 1

    best_pos, best_stress = None, None

    if effective_n_jobs(n_jobs) == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(
                dissimilarities, metric=metric,
                n_components=n_components, init=init,
                max_iter=max_iter, verbose=verbose,
                eps=eps, random_state=random_state)
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities, metric=metric, n_components=n_components,
                init=init, max_iter=max_iter, verbose=verbose, eps=eps,
                random_state=seed)
            for seed in seeds)
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress


class MDS(BaseEstimator):
    """Multidimensional scaling

    Read more in the :ref:`User Guide <multidimensional_scaling>`.

    Parameters
    ----------
    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities.

    metric : boolean, optional, default: True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.

    n_init : int, optional, default: 4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.

    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, optional, default: 0
        Level of verbosity.

    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional, default: None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    dissimilarity : 'euclidean' | 'precomputed', optional, default: 'euclidean'
        Dissimilarity measure to use:

        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.

        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.

    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.

    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import MDS
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = MDS(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)

    References
    ----------
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)

    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)

    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    """
    def __init__(self, n_components=2, metric=True, n_init=4,
                 max_iter=300, verbose=0, eps=1e-3, n_jobs=None,
                 random_state=None, dissimilarity="euclidean"):
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, init=None):
        """
        Computes the position of the points in the embedding space

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored

        init : ndarray, shape (n_samples,), optional, default: None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.
        """
        self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X, y=None, init=None):
        """
        Fit the data from X, and returns the embedded coordinates

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored

        init : ndarray, shape (n_samples,), optional, default: None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.
        """
        X = check_array(X)
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn("The MDS API has changed. ``fit`` now constructs an"
                          " dissimilarity matrix from data. To use a custom "
                          "dissimilarity matrix, set "
                          "``dissimilarity='precomputed'``.")

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)
        else:
            raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
                             " Got %s instead" % str(self.dissimilarity))

        self.embedding_, self.stress_, self.n_iter_ = smacof(
            self.dissimilarity_matrix_, metric=self.metric,
            n_components=self.n_components, init=init, n_init=self.n_init,
            n_jobs=self.n_jobs, max_iter=self.max_iter, verbose=self.verbose,
            eps=self.eps, random_state=self.random_state,
            return_n_iter=True)

        return self.embedding_

def demo():
    """# simple HOW-TO_USE the MDS in this file:

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(np.random.randint(0,1000,size=(500, 12)), columns=list('ABCDEFGHIJKL')) # <--- random array, but you would load your methylarray data here
    mds = MDS(n_jobs=-1, random_state=1, verbose=1)
    mds_transformed = mds.fit_transform(df.values)

    fig = plt.figure(figsize=(12, 9))
    plt.title('MDS Plot of betas from methylation data')
    plt.grid()
    plt.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=7, c='xkcd:ivory', edgecolor='black', linewidth='0.2',)
    plt.show()
    """
    pass
