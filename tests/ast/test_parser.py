import pimacs.ast.ast as ast
from pimacs.ast.parser import get_lark_parser, get_parser

parser = get_parser()


def test_attribute():
    code = "var a = obj.attr"
    nodes = parser.parser.parse(code)
    var_node = nodes[0]
    print(var_node)
    assert isinstance(var_node, ast.VarDecl)
    assert isinstance(var_node.init, ast.UAttr)

    attr_node = var_node.init
    assert attr_node.attr == "attr"

    assert isinstance(attr_node.value, ast.UVarRef)
    assert attr_node.value.name == "obj"


def test_nested_attribute():
    code = "var a = obj0.attr0.attr1"
    nodes = parser.parser.parse(code)
    var_node = nodes[0]
    print(var_node)

    assert isinstance(var_node, ast.VarDecl)
    assert isinstance(var_node.init, ast.UAttr)

    attr1 = var_node.init
    assert attr1.attr == "attr1"

    assert isinstance(attr1.value, ast.UAttr)
    attr0 = attr1.value
    assert attr0.attr == "attr0"


def test_call():
    code = "var a = some_fn(1, 2)"
    var_node = parser.parser.parse(code)[0]
    print(var_node)
    assert isinstance(var_node.init, ast.Call)
    func = var_node.init.target
    assert isinstance(func, ast.UFunction)

    code = "var a = %elisp_fn()"
    var_node = parser.parser.parse(code)[0]
    print(var_node)
    func = var_node.init.target
    assert isinstance(func, ast.UFunction)
    assert func.name == "%elisp_fn"


def test_class_method_call():
    code = "var a = obj.some_fn(1, 2)"
    var_node = parser.parser.parse(code)[0]
    print(var_node)
    call = var_node.init
    attr = call.target
    assert isinstance(attr, ast.UAttr)


def test_lisp_var_decl():
    code = "let begin :Int = %org-elemnt-property(%:contents-begin, self.elem)"
    var_node = parser.parser.parse(code)[0]
    print(var_node)


if __name__ == "__main__":
    # test_class_method_call()
    test_call()
