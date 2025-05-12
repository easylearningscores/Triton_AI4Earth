import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import _thread
from types import GenericAlias


__all__ = ['dataclass',
           'field',
           'Field',
           'FrozenInstanceError',
           'InitVar',
           'MISSING',

           # Helper functions.
           'fields',
           'asdict',
           'astuple',
           'make_dataclass',
           'replace',
           'is_dataclass',
           ]

class FrozenInstanceError(AttributeError): pass


class _HAS_DEFAULT_FACTORY_CLASS:
    def __repr__(self):
        return '<factory>'
_HAS_DEFAULT_FACTORY = _HAS_DEFAULT_FACTORY_CLASS()


class _MISSING_TYPE:
    pass
MISSING = _MISSING_TYPE()


_EMPTY_METADATA = types.MappingProxyType({})

class _FIELD_BASE:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
_FIELD = _FIELD_BASE('_FIELD')
_FIELD_CLASSVAR = _FIELD_BASE('_FIELD_CLASSVAR')
_FIELD_INITVAR = _FIELD_BASE('_FIELD_INITVAR')


_FIELDS = '__dataclass_fields__'


_PARAMS = '__dataclass_params__'


_POST_INIT_NAME = '__post_init__'


_MODULE_IDENTIFIER_RE = re.compile(r'^(?:\s*(\w+)\s*\.)?\s*(\w+)')

class InitVar:
    __slots__ = ('type', )

    def __init__(self, type):
        self.type = type

    def __repr__(self):
        if isinstance(self.type, type):
            type_name = self.type.__name__
        else:
            # typing objects, e.g. List[int]
            type_name = repr(self.type)
        return f'dataclasses.InitVar[{type_name}]'

    def __class_getitem__(cls, type):
        return InitVar(type)


class Field:
    __slots__ = ('name',
                 'type',
                 'default',
                 'default_factory',
                 'repr',
                 'hash',
                 'init',
                 'compare',
                 'metadata',
                 '_field_type',  # Private: not to be used by user code.
                 )

    def __init__(self, default, default_factory, init, repr, hash, compare,
                 metadata):
        self.name = None
        self.type = None
        self.default = default
        self.default_factory = default_factory
        self.init = init
        self.repr = repr
        self.hash = hash
        self.compare = compare
        self.metadata = (_EMPTY_METADATA
                         if metadata is None else
                         types.MappingProxyType(metadata))
        self._field_type = None

    def __repr__(self):
        return ('Field('
                f'name={self.name!r},'
                f'type={self.type!r},'
                f'default={self.default!r},'
                f'default_factory={self.default_factory!r},'
                f'init={self.init!r},'
                f'repr={self.repr!r},'
                f'hash={self.hash!r},'
                f'compare={self.compare!r},'
                f'metadata={self.metadata!r},'
                f'_field_type={self._field_type}'
                ')')

    def __set_name__(self, owner, name):
        func = getattr(type(self.default), '__set_name__', None)
        if func:
           
            func(self.default, owner, name)

    __class_getitem__ = classmethod(GenericAlias)


class _DataclassParams:
    __slots__ = ('init',
                 'repr',
                 'eq',
                 'order',
                 'unsafe_hash',
                 'frozen',
                 )

    def __init__(self, init, repr, eq, order, unsafe_hash, frozen):
        self.init = init
        self.repr = repr
        self.eq = eq
        self.order = order
        self.unsafe_hash = unsafe_hash
        self.frozen = frozen

    def __repr__(self):
        return ('_DataclassParams('
                f'init={self.init!r},'
                f'repr={self.repr!r},'
                f'eq={self.eq!r},'
                f'order={self.order!r},'
                f'unsafe_hash={self.unsafe_hash!r},'
                f'frozen={self.frozen!r}'
                ')')


def field(*, default=MISSING, default_factory=MISSING, init=True, repr=True,
          hash=None, compare=True, metadata=None):
    

    if default is not MISSING and default_factory is not MISSING:
        raise ValueError('cannot specify both default and default_factory')
    return Field(default, default_factory, init, repr, hash, compare,
                 metadata)


def _tuple_str(obj_name, fields):

    if not fields:
        return '()'
    return f'({",".join([f"{obj_name}.{f.name}" for f in fields])},)'


def _recursive_repr(user_function):

    repr_running = set()

    @functools.wraps(user_function)
    def wrapper(self):
        key = id(self), _thread.get_ident()
        if key in repr_running:
            return '...'
        repr_running.add(key)
        try:
            result = user_function(self)
        finally:
            repr_running.discard(key)
        return result
    return wrapper


def _create_fn(name, args, body, *, globals=None, locals=None,
               return_type=MISSING):

    if locals is None:
        locals = {}
    if 'BUILTINS' not in locals:
        locals['BUILTINS'] = builtins
    return_annotation = ''
    if return_type is not MISSING:
        locals['_return_type'] = return_type
        return_annotation = '->_return_type'
    args = ','.join(args)
    body = '\n'.join(f'  {b}' for b in body)

    txt = f' def {name}({args}){return_annotation}:\n{body}'

    local_vars = ', '.join(locals.keys())
    txt = f"def __create_fn__({local_vars}):\n{txt}\n return {name}"

    ns = {}
    exec(txt, globals, ns)
    return ns['__create_fn__'](**locals)


def _field_assign(frozen, name, value, self_name):

    if frozen:
        return f'BUILTINS.object.__setattr__({self_name},{name!r},{value})'
    return f'{self_name}.{name}={value}'


def _field_init(f, frozen, globals, self_name):


    default_name = f'_dflt_{f.name}'
    if f.default_factory is not MISSING:
        if f.init:

            globals[default_name] = f.default_factory
            value = (f'{default_name}() '
                     f'if {f.name} is _HAS_DEFAULT_FACTORY '
                     f'else {f.name}')
        else:
            

            globals[default_name] = f.default_factory
            value = f'{default_name}()'
    else:
        if f.init:
            if f.default is MISSING:
                value = f.name
            elif f.default is not MISSING:
                globals[default_name] = f.default
                value = f.name
        else:
     
            return None

    if f._field_type is _FIELD_INITVAR:
        return None

    return _field_assign(frozen, f.name, value, self_name)


def _init_param(f):

    if f.default is MISSING and f.default_factory is MISSING:

        default = ''
    elif f.default is not MISSING:

        default = f'=_dflt_{f.name}'
    elif f.default_factory is not MISSING:
        default = '=_HAS_DEFAULT_FACTORY'
    return f'{f.name}:_type_{f.name}{default}'


def _init_fn(fields, frozen, has_post_init, self_name, globals):

    seen_default = False
    for f in fields:
        if f.init:
            if not (f.default is MISSING and f.default_factory is MISSING):
                seen_default = True
            elif seen_default:
                raise TypeError(f'non-default argument {f.name!r} '
                                'follows default argument')

    locals = {f'_type_{f.name}': f.type for f in fields}
    locals.update({
        'MISSING': MISSING,
        '_HAS_DEFAULT_FACTORY': _HAS_DEFAULT_FACTORY,
    })

    body_lines = []
    for f in fields:
        line = _field_init(f, frozen, locals, self_name)
       
        if line:
            body_lines.append(line)

    if has_post_init:
        params_str = ','.join(f.name for f in fields
                              if f._field_type is _FIELD_INITVAR)
        body_lines.append(f'{self_name}.{_POST_INIT_NAME}({params_str})')

    # If no body lines, use 'pass'.
    if not body_lines:
        body_lines = ['pass']

    return _create_fn('__init__',
                      [self_name] + [_init_param(f) for f in fields if f.init],
                      body_lines,
                      locals=locals,
                      globals=globals,
                      return_type=None)


def _repr_fn(fields, globals):
    fn = _create_fn('__repr__',
                    ('self',),
                    ['return self.__class__.__qualname__ + f"(' +
                     ', '.join([f"{f.name}={{self.{f.name}!r}}"
                                for f in fields]) +
                     ')"'],
                     globals=globals)
    return _recursive_repr(fn)


def _frozen_get_del_attr(cls, fields, globals):
    locals = {'cls': cls,
              'FrozenInstanceError': FrozenInstanceError}
    if fields:
        fields_str = '(' + ','.join(repr(f.name) for f in fields) + ',)'
    else:
        fields_str = '()'
    return (_create_fn('__setattr__',
                      ('self', 'name', 'value'),
                      (f'if type(self) is cls or name in {fields_str}:',
                        ' raise FrozenInstanceError(f"cannot assign to field {name!r}")',
                       f'super(cls, self).__setattr__(name, value)'),
                       locals=locals,
                       globals=globals),
            _create_fn('__delattr__',
                      ('self', 'name'),
                      (f'if type(self) is cls or name in {fields_str}:',
                        ' raise FrozenInstanceError(f"cannot delete field {name!r}")',
                       f'super(cls, self).__delattr__(name)'),
                       locals=locals,
                       globals=globals),
            )


def _cmp_fn(name, op, self_tuple, other_tuple, globals):


    return _create_fn(name,
                      ('self', 'other'),
                      [ 'if other.__class__ is self.__class__:',
                       f' return {self_tuple}{op}{other_tuple}',
                        'return NotImplemented'],
                      globals=globals)


def _hash_fn(fields, globals):
    self_tuple = _tuple_str('self', fields)
    return _create_fn('__hash__',
                      ('self',),
                      [f'return hash({self_tuple})'],
                      globals=globals)


def _is_classvar(a_type, typing):

    return (a_type is typing.ClassVar
            or (type(a_type) is typing._GenericAlias
                and a_type.__origin__ is typing.ClassVar))


def _is_initvar(a_type, dataclasses):

    return (a_type is dataclasses.InitVar
            or type(a_type) is dataclasses.InitVar)


def _is_type(annotation, cls, a_module, a_type, is_type_predicate):
    

    match = _MODULE_IDENTIFIER_RE.match(annotation)
    if match:
        ns = None
        module_name = match.group(1)
        if not module_name:
 
            ns = sys.modules.get(cls.__module__).__dict__
        else:
            module = sys.modules.get(cls.__module__)
            if module and module.__dict__.get(module_name) is a_module:
                ns = sys.modules.get(a_type.__module__).__dict__
        if ns and is_type_predicate(ns.get(match.group(2)), a_module):
            return True
    return False


def _get_field(cls, a_name, a_type):
    
    default = getattr(cls, a_name, MISSING)
    if isinstance(default, Field):
        f = default
    else:
        if isinstance(default, types.MemberDescriptorType):
            default = MISSING
        f = field(default=default)

    f.name = a_name
    f.type = a_type

   
    f._field_type = _FIELD

   
    typing = sys.modules.get('typing')
    if typing:
        if (_is_classvar(a_type, typing)
            or (isinstance(f.type, str)
                and _is_type(f.type, cls, typing, typing.ClassVar,
                             _is_classvar))):
            f._field_type = _FIELD_CLASSVAR


    if f._field_type is _FIELD:

        dataclasses = sys.modules[__name__]
        if (_is_initvar(a_type, dataclasses)
            or (isinstance(f.type, str)
                and _is_type(f.type, cls, dataclasses, dataclasses.InitVar,
                             _is_initvar))):
            f._field_type = _FIELD_INITVAR


    if f._field_type in (_FIELD_CLASSVAR, _FIELD_INITVAR):
        if f.default_factory is not MISSING:
            raise TypeError(f'field {f.name} cannot have a '
                            'default factory')

    if f._field_type is _FIELD and isinstance(f.default, (list, dict, set)):
        raise ValueError(f'mutable default {type(f.default)} for field '
                         f'{f.name} is not allowed: use default_factory')

    return f


def _set_new_attribute(cls, name, value):

    if name in cls.__dict__:
        return True
    setattr(cls, name, value)
    return False


def _hash_set_none(cls, fields, globals):
    return None

def _hash_add(cls, fields, globals):
    flds = [f for f in fields if (f.compare if f.hash is None else f.hash)]
    return _hash_fn(flds, globals)

def _hash_exception(cls, fields, globals):
    # Raise an exception.
    raise TypeError(f'Cannot overwrite attribute __hash__ '
                    f'in class {cls.__name__}')


_hash_action = {(False, False, False, False): None,
                (False, False, False, True ): None,
                (False, False, True,  False): None,
                (False, False, True,  True ): None,
                (False, True,  False, False): _hash_set_none,
                (False, True,  False, True ): None,
                (False, True,  True,  False): _hash_add,
                (False, True,  True,  True ): None,
                (True,  False, False, False): _hash_add,
                (True,  False, False, True ): _hash_exception,
                (True,  False, True,  False): _hash_add,
                (True,  False, True,  True ): _hash_exception,
                (True,  True,  False, False): _hash_add,
                (True,  True,  False, True ): _hash_exception,
                (True,  True,  True,  False): _hash_add,
                (True,  True,  True,  True ): _hash_exception,
                }



def _process_class(cls, init, repr, eq, order, unsafe_hash, frozen):

    fields = {}

    if cls.__module__ in sys.modules:
        globals = sys.modules[cls.__module__].__dict__
    else:

        globals = {}

    setattr(cls, _PARAMS, _DataclassParams(init, repr, eq, order,
                                           unsafe_hash, frozen))


    any_frozen_base = False
    has_dataclass_bases = False
    for b in cls.__mro__[-1:0:-1]:

        base_fields = getattr(b, _FIELDS, None)
        if base_fields is not None:
            has_dataclass_bases = True
            for f in base_fields.values():
                fields[f.name] = f
            if getattr(b, _PARAMS).frozen:
                any_frozen_base = True

    
    cls_annotations = cls.__dict__.get('__annotations__', {})


    cls_fields = [_get_field(cls, name, type)
                  for name, type in cls_annotations.items()]
    for f in cls_fields:
        fields[f.name] = f

       
        if isinstance(getattr(cls, f.name, None), Field):
            if f.default is MISSING:
               
                delattr(cls, f.name)
            else:
                setattr(cls, f.name, f.default)

    for name, value in cls.__dict__.items():
        if isinstance(value, Field) and not name in cls_annotations:
            raise TypeError(f'{name!r} is a field but has no type annotation')

    if has_dataclass_bases:
        if any_frozen_base and not frozen:
            raise TypeError('cannot inherit non-frozen dataclass from a '
                            'frozen one')

        if not any_frozen_base and frozen:
            raise TypeError('cannot inherit frozen dataclass from a '
                            'non-frozen one')

 
    setattr(cls, _FIELDS, fields)


    class_hash = cls.__dict__.get('__hash__', MISSING)
    has_explicit_hash = not (class_hash is MISSING or
                             (class_hash is None and '__eq__' in cls.__dict__))

    if order and not eq:
        raise ValueError('eq must be true if order is true')

    if init:
        has_post_init = hasattr(cls, _POST_INIT_NAME)

        flds = [f for f in fields.values()
                if f._field_type in (_FIELD, _FIELD_INITVAR)]
        _set_new_attribute(cls, '__init__',
                           _init_fn(flds,
                                    frozen,
                                    has_post_init,
                                   
                                    '__dataclass_self__' if 'self' in fields
                                            else 'self',
                                    globals,
                          ))

    field_list = [f for f in fields.values() if f._field_type is _FIELD]

    if repr:
        flds = [f for f in field_list if f.repr]
        _set_new_attribute(cls, '__repr__', _repr_fn(flds, globals))

    if eq:
       
        flds = [f for f in field_list if f.compare]
        self_tuple = _tuple_str('self', flds)
        other_tuple = _tuple_str('other', flds)
        _set_new_attribute(cls, '__eq__',
                           _cmp_fn('__eq__', '==',
                                   self_tuple, other_tuple,
                                   globals=globals))

    if order:
        flds = [f for f in field_list if f.compare]
        self_tuple = _tuple_str('self', flds)
        other_tuple = _tuple_str('other', flds)
        for name, op in [('__lt__', '<'),
                         ('__le__', '<='),
                         ('__gt__', '>'),
                         ('__ge__', '>='),
                         ]:
            if _set_new_attribute(cls, name,
                                  _cmp_fn(name, op, self_tuple, other_tuple,
                                          globals=globals)):
                raise TypeError(f'Cannot overwrite attribute {name} '
                                f'in class {cls.__name__}. Consider using '
                                'functools.total_ordering')

    if frozen:
        for fn in _frozen_get_del_attr(cls, field_list, globals):
            if _set_new_attribute(cls, fn.__name__, fn):
                raise TypeError(f'Cannot overwrite attribute {fn.__name__} '
                                f'in class {cls.__name__}')

    hash_action = _hash_action[bool(unsafe_hash),
                               bool(eq),
                               bool(frozen),
                               has_explicit_hash]
    if hash_action:
  
        cls.__hash__ = hash_action(cls, field_list, globals)

    if not getattr(cls, '__doc__'):
        cls.__doc__ = (cls.__name__ +
                       str(inspect.signature(cls)).replace(' -> None', ''))

    return cls


def dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False,
              unsafe_hash=False, frozen=False):
   

    def wrap(cls):
        return _process_class(cls, init, repr, eq, order, unsafe_hash, frozen)

    if cls is None:
        return wrap

    return wrap(cls)


def fields(class_or_instance):
   
    try:
        fields = getattr(class_or_instance, _FIELDS)
    except AttributeError:
        raise TypeError('must be called with a dataclass type or instance')


    return tuple(f for f in fields.values() if f._field_type is _FIELD)


def _is_dataclass_instance(obj):
    """Returns True if obj is an instance of a dataclass."""
    return hasattr(type(obj), _FIELDS)


def is_dataclass(obj):
    """Returns True if obj is a dataclass or an instance of a
    dataclass."""
    cls = obj if isinstance(obj, type) else type(obj)
    return hasattr(cls, _FIELDS)


def asdict(obj, *, dict_factory=dict):
   
    if not _is_dataclass_instance(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory)


def _asdict_inner(obj, dict_factory):
    if _is_dataclass_instance(obj):
        result = []
        for f in fields(obj):
            value = _asdict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        

        return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        
        return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_asdict_inner(k, dict_factory),
                          _asdict_inner(v, dict_factory))
                         for k, v in obj.items())
    else:
        return copy.deepcopy(obj)


def astuple(obj, *, tuple_factory=tuple):
    

    if not _is_dataclass_instance(obj):
        raise TypeError("astuple() should be called on dataclass instances")
    return _astuple_inner(obj, tuple_factory)


def _astuple_inner(obj, tuple_factory):
    if _is_dataclass_instance(obj):
        result = []
        for f in fields(obj):
            value = _astuple_inner(getattr(obj, f.name), tuple_factory)
            result.append(value)
        return tuple_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
       
        return type(obj)(*[_astuple_inner(v, tuple_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
       
        return type(obj)(_astuple_inner(v, tuple_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_astuple_inner(k, tuple_factory), _astuple_inner(v, tuple_factory))
                          for k, v in obj.items())
    else:
        return copy.deepcopy(obj)


def make_dataclass(cls_name, fields, *, bases=(), namespace=None, init=True,
                   repr=True, eq=True, order=False, unsafe_hash=False,
                   frozen=False):
    

    if namespace is None:
        namespace = {}
    else:
        namespace = namespace.copy()

    seen = set()
    anns = {}
    for item in fields:
        if isinstance(item, str):
            name = item
            tp = 'typing.Any'
        elif len(item) == 2:
            name, tp, = item
        elif len(item) == 3:
            name, tp, spec = item
            namespace[name] = spec
        else:
            raise TypeError(f'Invalid field: {item!r}')

        if not isinstance(name, str) or not name.isidentifier():
            raise TypeError(f'Field names must be valid identifiers: {name!r}')
        if keyword.iskeyword(name):
            raise TypeError(f'Field names must not be keywords: {name!r}')
        if name in seen:
            raise TypeError(f'Field name duplicated: {name!r}')

        seen.add(name)
        anns[name] = tp

    namespace['__annotations__'] = anns
    
    cls = types.new_class(cls_name, bases, {}, lambda ns: ns.update(namespace))
    return dataclass(cls, init=init, repr=repr, eq=eq, order=order,
                     unsafe_hash=unsafe_hash, frozen=frozen)


def replace(obj, /, **changes):
   
    if not _is_dataclass_instance(obj):
        raise TypeError("replace() should be called on dataclass instances")


    for f in getattr(obj, _FIELDS).values():
        if f._field_type is _FIELD_CLASSVAR:
            continue

        if not f.init:
            # Error if this field is specified in changes.
            if f.name in changes:
                raise ValueError(f'field {f.name} is declared with '
                                 'init=False, it cannot be specified with '
                                 'replace()')
            continue

        if f.name not in changes:
            if f._field_type is _FIELD_INITVAR and f.default is MISSING:
                raise ValueError(f"InitVar {f.name!r} "
                                 'must be specified with replace()')
            changes[f.name] = getattr(obj, f.name)


    return obj.__class__(**changes)

from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


import math
from typing import List, Optional, Tuple

import torch
import torch.fft
import torch.onnx
from torch import Tensor
from torch.autograd import Function



def rfft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    
    if not torch.onnx.is_in_onnx_export():
        return torch.fft.rfft(input, n=n, dim=dim, norm=norm)

    if not isinstance(dim, int):
        raise TypeError()
    return _rfft_onnx(input, (n,), (dim,), norm)


def rfft2(
    input: Tensor,
    s: Optional[Tuple[int]] = None,
    dim: Tuple[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
   
    if not torch.onnx.is_in_onnx_export():
        return torch.fft.rfft2(input, s=s, dim=dim, norm=norm)

    if not (isinstance(dim, tuple) and len(dim) == 2):
        raise ValueError()
    return _rfft_onnx(input, s, dim, norm)


def irfft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
   
    if not torch.onnx.is_in_onnx_export():
        return torch.fft.irfft(input, n=n, dim=dim, norm=norm)

    if not isinstance(dim, int):
        raise TypeError()
    return _irfft_onnx(input, (n,), (dim,), norm)


def irfft2(
    input: Tensor,
    s: Optional[Tuple[int]] = None,
    dim: Tuple[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    
    if not torch.onnx.is_in_onnx_export():
        return torch.fft.irfft2(input, s=s, dim=dim, norm=norm)

    if not (isinstance(dim, tuple) and len(dim) == 2):
        raise ValueError()
    return _irfft_onnx(input, s, dim, norm)


def view_as_complex(input: Tensor) -> Tensor:
    
    if not torch.onnx.is_in_onnx_export():
        return torch.view_as_complex(input)

    # Just return the input unchanged - during ONNX export
    # there will be no complex type.
    if input.size(-1) != 2:
        raise ValueError
    return input


def real(input: Tensor) -> Tensor:
    
    if not torch.onnx.is_in_onnx_export():
        return input.real

    
    if input.size(-1) != 2:
        raise ValueError()
    return input[..., 0]


def imag(input: Tensor) -> Tensor:
   
    if not torch.onnx.is_in_onnx_export():
        return input.imag

    
    if input.size(-1) != 2:
        raise ValueError(input.size(-1))
    return input[..., 1]


def _rfft_onnx(
    input: Tensor, s: Optional[Tuple[Optional[int]]], dim: Tuple[int], norm: str
) -> Tensor:
    if s is not None:
        _check_padding_rfft(s, dim, input.size())

    ndim = len(dim)
    if ndim not in [1, 2]:
        raise ValueError(ndim)

    perm = not _is_last_dims(dim, input.ndim)

    if perm:
        perm_in, perm_out = _create_axes_perm(input.ndim, dim)
        # Add a dimension to account for complex output.
        perm_out.append(len(perm_out))
        # Transpose -> RFFT -> Transpose (inverse).
        input = input.permute(perm_in)

    rfft_func = OnnxRfft if ndim == 1 else OnnxRfft2
    output = rfft_func.apply(input)

    output = _scale_output_forward(output, norm, input.size(), ndim)

    if perm:
        output = output.permute(perm_out)

    return output


def _irfft_onnx(
    input: Tensor, s: Optional[Tuple[Optional[int]]], dim: Tuple[int], norm: str
) -> Tensor:
    if s is not None:
        _check_padding_irfft(s, dim, input.size())

    ndim = len(dim)
    if ndim not in [1, 2]:
        raise ValueError(ndim)

    # Whether to permute axes when DFT axis is not the last.
    perm = not _is_last_dims(dim, input.ndim)

    if perm:
        # Do not include last dimension (input is complex).
        perm_in, perm_out = _create_axes_perm(input.ndim - 1, dim)
        # Add a dimension to account for complex input.
        perm_in.append(len(perm_in))
        # Transpose -> IRFFT -> Transpose (inverse).
        input = input.permute(perm_in)

    irfft_func = OnnxIrfft if ndim == 1 else OnnxIrfft2
    output = irfft_func.apply(input)

    output = _scale_output_backward(output, norm, input.size(), ndim)

    if perm:
        output = output.permute(perm_out)

    return output


def _contrib_rfft(g: torch.Graph, input: torch.Value, ndim: int) -> torch.Value:
    if ndim not in [1, 2]:
        raise ValueError(ndim)

    output = g.op(
        "com.microsoft::Rfft",
        input,
        normalized_i=0,
        onesided_i=1,
        signal_ndim_i=ndim,
    )

    return output


def _contrib_irfft(g: torch.Graph, input: torch.Value, ndim: int) -> torch.Value:
    if ndim not in [1, 2]:
        raise ValueError(ndim)

    output = g.op(
        "com.microsoft::Irfft",
        input,
        normalized_i=0,
        onesided_i=1,
        signal_ndim_i=ndim,
    )

    return output


def _is_last_dims(dim: Tuple[int], inp_ndim: int) -> bool:
    ndim = len(dim)
    for i, idim in enumerate(dim):
        # This takes care of both positive and negative axis indices.
        if idim % inp_ndim != inp_ndim - ndim + i:
            return False
    return True


def _check_padding_rfft(
    sizes: Tuple[Optional[int]], dim: Tuple[int], inp_sizes: Tuple[int]
) -> None:
    if len(sizes) != len(dim):
        raise ValueError(f"{sizes}, {dim}")
    for i, s in enumerate(sizes):
        if s is None or s < 0:
            continue
        # Current Contrib RFFT does not support pad/trim yet.
        if s != inp_sizes[dim[i]]:
            raise RuntimeError(
                f"Padding/trimming is not yet supported, "
                f"got sizes {sizes}, DFT dims {dim}, "
                f"input dims {inp_sizes}."
            )


def _check_padding_irfft(
    sizes: Tuple[Optional[int]], dim: Tuple[int], inp_sizes: Tuple[int]
) -> None:
    if len(sizes) != len(dim):
        raise ValueError(f"{sizes}, {dim}")
    # All but last dims must be equal to input dims.
    for i, s in enumerate(sizes[:-1]):
        if s is None or s < 0:
            continue
        # Current Contrib RFFT does not support pad/trim yet.
        if s != inp_sizes[dim[i]]:
            raise RuntimeError(
                f"Padding/trimming is not yet supported, "
                f"got sizes {sizes}, DFT dims {dim}, "
                f"input dims {inp_sizes}."
            )
    # Check last dim.
    s = sizes[-1]
    if s is not None and s > 0:
        expected_size = 2 * (inp_sizes[dim[-1]] - 1)
        if s != expected_size:
            raise RuntimeError(
                f"Padding/trimming is not yet supported, got sizes {sizes}"
                f", DFT dims {dim}, input dims {inp_sizes}"
                f", expected last size {expected_size}."
            )


def _create_axes_perm(ndim: int, dims: Tuple[int]) -> Tuple[List[int], List[int]]:
    """Creates permuted axes indices for RFFT/IRFFT operators."""
    perm_in = list(range(ndim))
    perm_out = list(perm_in)
    # Move indices to the right to make 'dims' as innermost dimensions.
    for i in range(-1, -(len(dims) + 1), -1):
        perm_in[dims[i]], perm_in[i] = perm_in[i], perm_in[dims[i]]
    # Move indices to the left to restore original shape.
    for i in range(-len(dims), 0):
        perm_out[dims[i]], perm_out[i] = perm_out[i], perm_out[dims[i]]

    return perm_in, perm_out


def _scale_output_forward(
    output: Tensor, norm: str, sizes: torch.Size, ndim: int
) -> Tensor:
    """Scales the RFFT output according to norm parameter."""

    norm = "backward" if norm is None else norm
    if norm not in ["forward", "backward", "ortho"]:
        raise ValueError(norm)

    if norm in ["forward", "ortho"]:
        
        dft_size = math.prod(sizes[-ndim:]).float()
        denom = torch.sqrt(dft_size) if norm == "ortho" else dft_size
        output = output / denom

    return output


def _scale_output_backward(
    output: Tensor, norm: str, sizes: torch.Size, ndim: int
) -> Tensor:
    """Scales the IRFFT output according to norm parameter."""

    norm = "backward" if norm is None else norm
    if norm not in ["forward", "backward", "ortho"]:
        raise ValueError(norm)

    
    if norm in ["forward", "ortho"]:
       
        if not len(sizes) >= ndim + 1:
            raise ValueError
        dft_size = math.prod(sizes[-(ndim + 1) : -2])
        dft_size *= 2 * (sizes[-2] - 1)
        dft_size = dft_size.float()
        # Since cuFFT scales by 1/dft_size, replace this scale with appropriate one.
        scale = dft_size if norm == "forward" else torch.sqrt(dft_size)
        output = scale * output

    return output


class OnnxRfft(Function):
   

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            raise ValueError("Must be called only during ONNX export.")


        y = torch.fft.rfft(input, dim=-1, norm="backward")
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph"""
        return _contrib_rfft(g, input, ndim=1)


class OnnxRfft2(Function):
   

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            raise AssertionError("Must be called only during ONNX export.")

        y = torch.fft.rfft2(input, dim=(-2, -1), norm="backward")
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph"""
        return _contrib_rfft(g, input, ndim=2)


class OnnxIrfft(Function):
   

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            raise ValueError("Must be called only during ONNX export.")

       
        return torch.fft.irfft(torch.view_as_complex(input), dim=-1, norm="backward")

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph"""
        return _contrib_irfft(g, input, ndim=1)


class OnnxIrfft2(Function):
    
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            raise AssertionError("Must be called only during ONNX export.")

        
        return torch.fft.irfft2(
            torch.view_as_complex(input), dim=(-2, -1), norm="backward"
        )

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph"""
        return _contrib_irfft(g, input, ndim=2)



@dataclass
class ModelMetaData:
    

    # Model info
    name: str = "ModulusModule"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    amp_cpu: bool = None
    amp_gpu: bool = None
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    onnx_gpu: bool = None
    onnx_cpu: bool = None
    onnx_runtime: bool = False
    trt: bool = False
    # Physics informed
    var_dim: int = -1
    func_torch: bool = False
    auto_grad: bool = False

    def __post_init__(self):
        self.amp_cpu = self.amp if self.amp_cpu is None else self.amp_cpu
        self.amp_gpu = self.amp if self.amp_gpu is None else self.amp_gpu
        self.onnx_cpu = self.onnx if self.onnx_cpu is None else self.onnx_cpu
        self.onnx_gpu = self.onnx if self.onnx_gpu is None else self.onnx_gpu


import importlib
import inspect
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import torch


@dataclass
class ModelMetaData:
    """Data class for storing essential meta data needed for all Modulus Models"""

    # Model info
    name: str = "ModulusModule"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    amp_cpu: bool = None
    amp_gpu: bool = None
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    onnx_gpu: bool = None
    onnx_cpu: bool = None
    onnx_runtime: bool = False
    trt: bool = False
    # Physics informed
    var_dim: int = -1
    func_torch: bool = False
    auto_grad: bool = False

    def __post_init__(self):
        self.amp_cpu = self.amp if self.amp_cpu is None else self.amp_cpu
        self.amp_gpu = self.amp if self.amp_gpu is None else self.amp_gpu
        self.onnx_cpu = self.onnx if self.onnx_cpu is None else self.onnx_cpu
        self.onnx_gpu = self.onnx if self.onnx_gpu is None else self.onnx_gpu



from importlib.metadata import EntryPoint, entry_points
from typing import List, Union

# This import is required for compatibility with doctests.
import importlib_metadata




class ModelRegistry:
    _shared_state = {"_model_registry": None}

    def __new__(cls, *args, **kwargs):
        obj = super(ModelRegistry, cls).__new__(cls)
        obj.__dict__ = cls._shared_state
        if cls._shared_state["_model_registry"] is None:
            cls._shared_state["_model_registry"] = cls._construct_registry()
        return obj

    @staticmethod
    def _construct_registry() -> dict:
        registry = {}
        entrypoints = entry_points(group="modulus.models")
        for entry_point in entrypoints:
            registry[entry_point.name] = entry_point
        return registry

    def register(self, model: "modulus.Module", name: Union[str, None] = None) -> None:
    

        # Check if model is a modulus model
        if not issubclass(model, modulus.Module):
            raise ValueError(
                f"Only subclasses of modulus.Module can be registered. "
                f"Provided model is of type {type(model)}"
            )

        # If no name provided, use the model's name
        if name is None:
            name = model.__name__

        # Check if name already in use
        if name in self._model_registry:
            raise ValueError(f"Name {name} already in use")

        # Add this class to the dict of model registry
        self._model_registry[name] = model

    def factory(self, name: str) -> "modulus.Module":
       

        model = self._model_registry.get(name)
        if model is not None:
            if isinstance(model, (EntryPoint, importlib_metadata.EntryPoint)):
                model = model.load()
            return model

        raise KeyError(f"No model is registered under the name {name}")

    def list_models(self) -> List[str]:
       
        return list(self._model_registry.keys())

    def __clear_registry__(self):
        # NOTE: This is only used for testing purposes
        self._model_registry = {}

    def __restore_registry__(self):
        # NOTE: This is only used for testing purposes
        self._model_registry = self._construct_registry()



import hashlib
import json
import logging
import os
import re
import urllib.request
import zipfile

import fsspec
import fsspec.implementations.cached
import requests
import s3fs
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    LOCAL_CACHE = os.environ["LOCAL_CACHE"]
except KeyError:
    LOCAL_CACHE = os.environ["HOME"] + "/.cache/modulus"


def _cache_fs(fs):
    return fsspec.implementations.cached.CachingFileSystem(
        fs=fs, cache_storage=LOCAL_CACHE
    )


def _get_fs(path):
    if path.startswith("s3://"):
        return s3fs.S3FileSystem(client_kwargs=dict(endpoint_url="https://pbss.s8k.io"))
    else:
        return fsspec.filesystem("file")


def _download_ngc_model_file(path: str, out_path: str, timeout: int = 300) -> str:
    
    # Strip ngc model url prefix
    suffix = "ngc://models/"
    # The regex check
    pattern = re.compile(f"{suffix}[\w-]+(/[\w-]+)?/[\w-]+@[A-Za-z0-9.]+/[\w/](.*)")
    if not pattern.match(path):
        raise ValueError(
            "Invalid URL, should be of form ngc://models/<org_id/team_id/model_id>@<version>/<path/in/repo>"
        )

    path = path.replace(suffix, "")
    if len(path.split("@")[0].split("/")) == 3:
        (org, team, model_version, filename) = path.split("/", 3)
        (model, version) = model_version.split("@", 1)
    else:
        (org, model_version, filename) = path.split("/", 2)
        (model, version) = model_version.split("@", 1)
        team = None

    token = ""
    # If API key environment variable
    if "NGC_API_KEY" in os.environ:
        try:
            # SSA tokens
            if os.environ["NGC_API_KEY"].startswith("nvapi-"):
                raise NotImplementedError("New personal keys not supported yet")
            # Legacy tokens
            # https://docs.nvidia.com/ngc/gpu-cloud/ngc-catalog-user-guide/index.html#download-models-via-wget-authenticated-access
            else:
                session = requests.Session()
                session.auth = ("$oauthtoken", os.environ["NGC_API_KEY"])
                headers = {"Accept": "application/json"}
                authn_url = f"https://authn.nvidia.com/token?service=ngc&scope=group/ngc:{org}&group/ngc:{org}/{team}"
                r = session.get(authn_url, headers=headers, timeout=5)
                r.raise_for_status()
                token = json.loads(r.content)["token"]
        except requests.exceptions.RequestException:
            logger.warning(
                "Failed to get JWT using the API set in NGC_API_KEY environment variable"
            )
            raise  # Re-raise the exception

    # Download file, apparently the URL for private registries is different than the public?
    if len(token) > 0:
        # Sloppy but works
        if team:
            file_url = f"https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/models/{model}/versions/{version}/files/{filename}"
        else:
            file_url = f"https://api.ngc.nvidia.com/v2/org/{org}/models/{model}/versions/{version}/files/{filename}"
    else:
        if team:
            file_url = f"https://api.ngc.nvidia.com/v2/models/{org}/{team}/{model}/versions/{version}/files/{filename}"
        else:
            file_url = f"https://api.ngc.nvidia.com/v2/models/{org}/{model}/versions/{version}/files/{filename}"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # Streaming here for larger files
    with requests.get(file_url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        chunk_size = 1024  # 1 kb
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        progress_bar.set_description(f"Fetching {filename}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()

    # Unzip contents if zip file (most model files are)
    if zipfile.is_zipfile(out_path) and path.endswith(".zip"):
        temp_path = out_path + ".zip"
        os.rename(out_path, temp_path)
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            zip_ref.extractall(out_path)
        # Clean up zip
        os.remove(temp_path)

    return out_path


def _download_cached(
    path: str, recursive: bool = False, local_cache_path: str = LOCAL_CACHE
) -> str:
    sha = hashlib.sha256(path.encode())
    filename = sha.hexdigest()
    try:
        os.makedirs(local_cache_path, exist_ok=True)
    except PermissionError as error:
        logger.error(
            "Failed to create cache folder, check permissions or set a cache"
            + " location using the LOCAL_CACHE environment variable"
        )
        raise error
    except OSError as error:
        logger.error(
            "Failed to create cache folder, set a cache"
            + " location using the LOCAL_CACHE environment variable"
        )
        raise error

    cache_path = os.path.join(local_cache_path, filename)

    url = urllib.parse.urlparse(path)

    # TODO watch for race condition here
    if not os.path.exists(cache_path):
        logger.debug("Downloading %s to cache: %s", path, cache_path)
        if path.startswith("s3://"):
            fs = _get_fs(path)
            fs.get(path, cache_path, recursive=recursive)
        elif path.startswith("ngc://models/"):
            path = _download_ngc_model_file(path, cache_path)
            return path
        elif url.scheme == "http":
            # urllib.request.urlretrieve(path, cache_path)
            # TODO: Check if this supports directory fetches
            response = requests.get(path, stream=True, timeout=5)
            with open(cache_path, "wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output.write(chunk)
        elif url.scheme == "file":
            path = os.path.join(url.netloc, url.path)
            return path
        else:
            return path

    else:
        logger.debug("Opening from cache: %s", cache_path)

    return cache_path


class Package:
 

    def __init__(self, root: str, seperator: str = "/"):
        self.root = root
        self.seperator = seperator

    def get(self, path: str, recursive: bool = False) -> str:
        """Get a local path to the item at ``path``

        ``path`` might be a remote file, in which case it is downloaded to a
        local cache at $LOCAL_CACHE or $HOME/.cache/modulus first.
        """
        return _download_cached(self._fullpath(path), recursive=recursive)

    def _fullpath(self, path):
        return self.root + self.seperator + path


class Module(torch.nn.Module):
   

    _file_extension = ".mdlus"  # Set file extension for saving and loading
    __model_checkpoint_version__ = (
        "0.1.0"  # Used for file versioning and is not the same as modulus version
    )

    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls)

        # Get signature of __init__ function
        sig = inspect.signature(cls.__init__)

        # Bind args and kwargs to signature
        bound_args = sig.bind_partial(
            *([None] + list(args)), **kwargs
        )  # Add None to account for self
        bound_args.apply_defaults()

        # Get args and kwargs (excluding self and unroll kwargs)
        instantiate_args = {}
        for param, (k, v) in zip(sig.parameters.values(), bound_args.arguments.items()):
            # Skip self
            if k == "self":
                continue

            # Add args and kwargs to instantiate_args
            if param.kind == param.VAR_KEYWORD:
                instantiate_args.update(v)
            else:
                instantiate_args[k] = v

        # Store args needed for instantiation
        out._args = {
            "__name__": cls.__name__,
            "__module__": cls.__module__,
            "__args__": instantiate_args,
        }
        return out

    def __init__(self, meta: Union[ModelMetaData, None] = None):
        super().__init__()
        self.meta = meta
        self.register_buffer("device_buffer", torch.empty(0))
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("core.module")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    @staticmethod
    def _safe_members(tar, local_path):
        for member in tar.getmembers():
            if (
                ".." in member.name
                or os.path.isabs(member.name)
                or os.path.realpath(os.path.join(local_path, member.name)).startswith(
                    os.path.realpath(local_path)
                )
            ):
                yield member
            else:
                print(f"Skipping potentially malicious file: {member.name}")

    @classmethod
    def instantiate(cls, arg_dict: Dict[str, Any]) -> "Module":
    
        _cls_name = arg_dict["__name__"]
        registry = ModelRegistry()
        if cls.__name__ == arg_dict["__name__"]:  # If cls is the class
            _cls = cls
        elif _cls_name in registry.list_models():  # Built in registry
            _cls = registry.factory(_cls_name)
        else:
            try:
                # Otherwise, try to import the class
                _mod = importlib.import_module(arg_dict["__module__"])
                _cls = getattr(_mod, arg_dict["__name__"])
            except AttributeError:
                # Cross fingers and hope for the best (maybe the class name changed)
                _cls = cls
        return _cls(**arg_dict["__args__"])

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s - {self.meta.name}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # TODO: set up debug log
        # fh = logging.FileHandler(f'modulus-core-{self.meta.name}.log')

    def save(self, file_name: Union[str, None] = None, verbose: bool = False) -> None:
        

        if file_name is not None and not file_name.endswith(self._file_extension):
            raise ValueError(
                f"File name must end with {self._file_extension} extension"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            torch.save(self.state_dict(), local_path / "model.pt")

            with open(local_path / "args.json", "w") as f:
                json.dump(self._args, f)

            # Save the modulus version and git hash (if available)
            metadata_info = {
                "modulus_version": modulus.__version__,
                "mdlus_file_version": self.__model_checkpoint_version__,
            }

            if verbose:
                import git

                try:
                    repo = git.Repo(search_parent_directories=True)
                    metadata_info["git_hash"] = repo.head.object.hexsha
                except git.InvalidGitRepositoryError:
                    metadata_info["git_hash"] = None

            with open(local_path / "metadata.json", "w") as f:
                json.dump(metadata_info, f)

            # Once all files are saved, package them into a tar file
            with tarfile.open(local_path / "model.tar", "w") as tar:
                for file in local_path.iterdir():
                    tar.add(str(file), arcname=file.name)

            if file_name is None:
                file_name = self.meta.name + ".mdlus"

            # Save files to remote destination
            fs = _get_fs(file_name)
            fs.put(str(local_path / "model.tar"), file_name)

    @staticmethod
    def _check_checkpoint(local_path: str) -> bool:
        if not local_path.joinpath("args.json").exists():
            raise IOError("File 'args.json' not found in checkpoint")

        if not local_path.joinpath("metadata.json").exists():
            raise IOError("File 'metadata.json' not found in checkpoint")

        if not local_path.joinpath("model.pt").exists():
            raise IOError("Model weights 'model.pt' not found in checkpoint")

        # Check if the checkpoint version is compatible with the current version
        with open(local_path.joinpath("metadata.json"), "r") as f:
            metadata_info = json.load(f)
            if (
                metadata_info["mdlus_file_version"]
                != Module.__model_checkpoint_version__
            ):
                raise IOError(
                    f"Model checkpoint version {metadata_info['mdlus_file_version']} is not compatible with current version {Module.__version__}"
                )

    def load(
        self,
        file_name: str,
        map_location: Union[None, str, torch.device] = None,
        strict: bool = True,
    ) -> None:
        

        # Download and cache the checkpoint file if needed
        cached_file_name = _download_cached(file_name)

        # Use a temporary directory to extract the tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            # Open the tar file and extract its contents to the temporary directory
            with tarfile.open(cached_file_name, "r") as tar:
                tar.extractall(
                    path=local_path, members=list(Module._safe_members(tar, local_path))
                )

            # Check if the checkpoint is valid
            Module._check_checkpoint(local_path)

            # Load the model weights
            device = map_location if map_location is not None else self.device
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=device
            )
            self.load_state_dict(model_dict, strict=strict)

    @classmethod
    def from_checkpoint(cls, file_name: str) -> "Module":
       
        # Download and cache the checkpoint file if needed
        cached_file_name = _download_cached(file_name)

        # Use a temporary directory to extract the tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            # Open the tar file and extract its contents to the temporary directory
            with tarfile.open(cached_file_name, "r") as tar:
                tar.extractall(
                    path=local_path, members=list(cls._safe_members(tar, local_path))
                )

            # Check if the checkpoint is valid
            Module._check_checkpoint(local_path)

            # Load model arguments and instantiate the model
            with open(local_path.joinpath("args.json"), "r") as f:
                args = json.load(f)
            model = cls.instantiate(args)

            # Load the model weights
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=model.device
            )
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def from_torch(
        torch_model_class: torch.nn.Module, meta: ModelMetaData = None
    ) -> "Module":
        

        # Define an internal class as before
        class ModulusModel(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(meta=meta)
                self.inner_model = torch_model_class(*args, **kwargs)

            def forward(self, x):
                return self.inner_model(x)

        # Get the argument names and default values of the PyTorch model's init method
        init_argspec = inspect.getfullargspec(torch_model_class.__init__)
        model_argnames = init_argspec.args[1:]  # Exclude 'self'
        model_defaults = init_argspec.defaults or []
        defaults_dict = dict(
            zip(model_argnames[-len(model_defaults) :], model_defaults)
        )

        # Define the signature of new init
        params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        params += [
            inspect.Parameter(
                argname,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults_dict.get(argname, inspect.Parameter.empty),
            )
            for argname in model_argnames
        ]
        init_signature = inspect.Signature(params)

        # Replace ModulusModel.__init__ signature with new init signature
        ModulusModel.__init__.__signature__ = init_signature

        # Generate a unique name for the created class
        new_class_name = f"{torch_model_class.__name__}ModulusModel"
        ModulusModel.__name__ = new_class_name

        # Add this class to the dict of models classes
        registry = ModelRegistry()
        registry.register(ModulusModel, new_class_name)

        return ModulusModel

    @property
    def device(self) -> torch.device:
      
        return self.device_buffer.device

    def num_parameters(self) -> int:
        """Gets the number of learnable parameters"""
        count = 0
        for name, param in self.named_parameters():
            count += param.numel()
        return count

Tensor = torch.Tensor

import torch.fft

class AFNOMlp(nn.Module):
    

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        out_features: int,
        activation_fn: nn.Module = nn.GELU(),
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, latent_features)
        self.act = activation_fn
        self.fc2 = nn.Linear(latent_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2DLayer(nn.Module):
   

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        if not (hidden_size % num_blocks == 0):
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
            )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape
        # Using ONNX friendly FFT functions
        x = rfft2(x, dim=(1, 2), norm="ortho")
        x_real, x_imag = real(x), imag(x)
        x_real = x_real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = F.relu(
            torch.einsum(
                "nyxbi,bio->nyxbo",
                x_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[0],
            )
            - torch.einsum(
                "nyxbi,bio->nyxbo",
                x_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_imag[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = F.relu(
            torch.einsum(
                "nyxbi,bio->nyxbo",
                x_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[0],
            )
            + torch.einsum(
                "nyxbi,bio->nyxbo",
                x_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[1],
            )
            + self.b1[1]
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 0
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            - torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 1
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            + torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = view_as_complex(x)
        
        if torch.onnx.is_in_onnx_export():
            x = x.reshape(B, H, W // 2 + 1, C, 2)
        else:
            x = x.reshape(B, H, W // 2 + 1, C)
        # Using ONNX friendly FFT functions
        x = irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        activation_fn: nn.Module = nn.GELU(),
        norm_layer: nn.Module = nn.LayerNorm,
        double_skip: bool = True,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = AFNO2DLayer(
            embed_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        self.mlp = AFNOMlp(
            in_features=embed_dim,
            latent_features=mlp_latent_dim,
            out_features=embed_dim,
            activation_fn=activation_fn,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
   

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
    ):
        super().__init__()
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        num_patches = (inp_shape[1] // patch_size[1]) * (inp_shape[0] // patch_size[0])
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        if not (H == self.inp_shape[0] and W == self.inp_shape[1]):
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model ({self.inp_shape[0]}*{self.inp_shape[1]})."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@dataclass
class MetaData(ModelMetaData):
    name: str = "AFNO"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Fourcastnet(Module):

    def __init__(
        self,
        params,
        inp_shape: tuple = [120, 240],
        in_channels: int = 97,
        out_channels: int = 93,
        patch_size: List[int] = [2, 2], #origianl 8
        embed_dim: int = 256, #origianl 256
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ) -> None:
        super().__init__(meta=MetaData())
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        if not (
            inp_shape[0] % patch_size[0] == 0 and inp_shape[1] % patch_size[1] == 0
        ):
            raise ValueError(
                f"input shape {inp_shape} should be divisible by patch_size {patch_size}"
            )

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            inp_shape=inp_shape,
            in_channels=self.in_chans,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_blocks=self.num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # What is this for
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {"pos_embed", "cls_token"}

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass of core AFNO"""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.head(x)

        # Correct tensor shape back into [B, C, H, W]
        # [b h w (p1 p2 c_out)]
        out = x.view(list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1])
        # [b h w p1 p2 c_out]
        out = torch.permute(out, (0, 5, 1, 3, 2, 4))
        # [b c_out, h, p1, w, p2]
        out = out.reshape(list(out.shape[:2]) + [self.inp_shape[0], self.inp_shape[1]])
        # [b c_out, (h*p1), (w*p2)]
        return out

from thop import profile

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = Fourcastnet().to(device)

    input = torch.randn(1, 97, 120, 240).to(device)
    output = net(input)

    macs, params = profile(net, inputs=(input, ))
    
    print('macs: ', macs, 'params: ', params)
    print('macs: %.2f G, params: %.2f M' % (macs / 1000000000.0, params / 1000000.0))
    print(output.shape)
