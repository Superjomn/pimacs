from typing import Dict, List, Optional, Tuple


class Type:
    # NOTE: Type should not be modified, any variant should clone a new type.
    _instances: Dict[tuple, "Type"] = {}

    def __new__(cls, *args, **kwargs):
        if not cls is PlaceholderType:
            key = (cls, args, tuple(kwargs.items()))
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)
            return cls._instances[key]
        else:
            return super().__new__(cls)

    def __init__(self, name, parent=None, params: Optional[Tuple["Type", ...]] = None,
                 is_concrete=True, is_optional: bool = False, module=None):
        self.name = name
        self.params = params or tuple()
        self.parent = parent
        self.module = module
        self._is_concrete = is_concrete
        self._initialized = True
        self._is_optional = is_optional

    @property
    def is_concrete(self) -> bool:
        return self._is_concrete

    @property
    def is_optional(self) -> bool:
        return self._is_optional

    def get_nosugar_type(self):
        '''
        Get the type without sugar, such as optional.
        '''
        return Type(self.name, self.parent, self.params)

    def get_optional_type(self):
        if self.is_optional:
            return self
        return Type(self.name, self.parent, self.params,
                    is_concrete=self.is_concrete,
                    is_optional=True)

    def can_accept(self, other: "Type") -> bool:
        if self == other:
            return True
        if isinstance(other, Type) and other.parent is not None:
            return self.can_accept(other.parent)
        return False

    @classmethod
    def get_primitive(cls, name) -> Optional["Type"]:
        if name == "Int":
            return Int
        elif name == "Float":
            return Float
        elif name == "Bool":
            return Bool
        elif name == "Str":
            return Str
        return None

    def is_List(self) -> bool:
        return self.name == "List"

    def is_Set(self):
        return self.name == "Set"

    def is_Dict(self):
        return self.name == "Dict"

    def __str__(self):
        optional_repr = "?" if self.is_optional else ""
        module_repr = f"{self.module.name}." if self.module else ""
        return module_repr + self.name + (f"[{', '.join(str(p) for p in self.params)}]" if self.params else "") + optional_repr

    def __eq__(self, other) -> bool:
        if not isinstance(other, Type):
            return False
        return self.name == other.name and \
            self.params == other.params and \
            self.parent == other.parent and \
            self._is_concrete == other._is_concrete

    def __hash__(self):
        return hash((self.name, self.parent, self.params, self._is_concrete))


class BasicType(Type):
    def __init__(self, name, parent=None):
        super().__init__(name=name, parent=parent, is_concrete=True)

    def __repr__(self):
        return self.name + ("?" if self._is_optional else "")


class CompositeType(Type):
    def __init__(self, name, parent=None, params: Optional[Tuple[Type, ...]] = None, module=None):
        super().__init__(name, parent, params=params, module=module)

    def __hash__(self):
        return super().__hash__()

    @property
    def is_concrete(self):
        return all(param.is_concrete for param in self.params)

    def clone_with(self, *params):
        return type(self)(self.name, self.parent, params=params, module=self.module)

    def replace_with(self, mapping: Dict[Type, Type]) -> "CompositeType":
        '''
        Replace the type with the mapping and get a new type
        '''
        params = tuple(mapping.get(param, param) for param in self.params)

        the_params = []
        for param in params:
            if (not param.is_concrete) and not isinstance(param, PlaceholderType):
                assert isinstance(param, CompositeType)
                param = param.replace_with(mapping)
            the_params.append(param)

        return CompositeType(self.name, self.parent, params=tuple(the_params), module=self.module)

    def collect_placeholders(self) -> List["PlaceholderType"]:
        placeholders = []
        for param in self.params:
            if isinstance(param, PlaceholderType):
                placeholders.append(param)
            elif isinstance(param, CompositeType):
                placeholders.extend(param.collect_placeholders())
        return placeholders

    def __eq__(self, other):
        if not isinstance(other, Type):
            return False

        if self.name != other.name:
            return False

        if len(self.params) != len(other.params):
            return False

        # TODO: check if the placeholder is the same
        for p1, p2 in zip(self.params, other.params):
            if isinstance(p1, PlaceholderType) and isinstance(p2, PlaceholderType):
                continue
            else:
                return p1 == p2
        return True

    def __repr__(self):
        return super().__str__()


class PlaceholderType(Type):
    def __init__(self, name, parent=None):
        super().__init__(name, parent, is_concrete=False)

    def __repr__(self):
        return f"<P: {self.name}>" + ("?" if self._is_optional else "")

    def compatible_with(self, other):
        return True


class GenericType(Type):
    def __init__(self, name, parent=None, module=None):
        super().__init__(name, parent, module=module)

    def __repr__(self):
        return f"<G: {super().__str__()}>" + ("?" if self._is_optional else "")


class FunctionType(Type):
    def __init__(self, return_type, arg_types: List[Type], params: Optional[Tuple[Type, ...]] = None, module=None):
        super().__init__("Function", params=params, module=module)
        self.return_type = return_type
        self.arg_types = arg_types

    def __repr__(self):
        arg_types = ', '.join(repr(arg) for arg in self.arg_types)
        param_types = ', '.join(repr(param) for param in self.params)
        return f"({arg_types})[{param_types}] -> {self.return_type}"

# Basic types


class NumberType(BasicType):
    def __init__(self, parent=None):
        super().__init__(name="Number", parent=parent)


class IntType(BasicType):
    def __init__(self):
        super().__init__("Int", parent=NumberType())


class FloatType(BasicType):
    def __init__(self):
        super().__init__("Float", parent=NumberType())


class BoolType(BasicType):
    def __init__(self):
        super().__init__("Bool")


class StrType(BasicType):
    def __init__(self):
        super().__init__("Str")


class NilType(BasicType):
    def __init__(self):
        super().__init__("Nil")


class UnkType(BasicType):
    def __init__(self):
        super().__init__("Unk")


class LispType_(BasicType):
    def __init__(self):
        super().__init__("Lisp")


# basic types
Number = NumberType()
Int = IntType()
Float = FloatType()
Bool = BoolType()
Str = StrType()
Nil = NilType()
Unk = UnkType()
LispType = LispType_()
ModuleType = BasicType("Module")

# container
Set_ = CompositeType("Set", parent=GenericType, params=(PlaceholderType("T"),))
List_ = CompositeType("List", parent=GenericType,
                      params=(PlaceholderType("T"),))
Dict_ = CompositeType("Dict", parent=GenericType, params=(
    PlaceholderType("K"), PlaceholderType("V")))


def get_resultant_type(type1, type2):
    if type1 == Float and type1 == Int:
        return Float
    elif type1 == Int and type2 == Float:
        return Float
    elif type1 == type2:
        return type1
    else:
        return Unk


def get_type(name: str, params: Optional[Tuple[Type, ...]] = None, module=None):
    if params:
        return CompositeType(name, params=params, module=module)
    return GenericType(name, module=module)
