import pimacs.ast.type as ty


def test_BasicType():
    assert ty.Int.parent is ty.Number
    assert ty.Float.parent is ty.Number


def test_CompositeType():
    T0 = ty.PlaceholderType("T0")
    T1 = ty.PlaceholderType("T1")
    AppType = ty.CompositeType("App", parent=ty.GenericType, params=(T0, T1))
    assert AppType.is_concrete == False

    AppType_Spec0 = AppType.clone_with(ty.Int, ty.Float)
    assert AppType_Spec0.is_concrete == True

    AppType_Spec1 = AppType_Spec0.clone_with(ty.Float, T0)
    assert AppType_Spec1.is_concrete == False


def test_CompositeType_unique():
    T0 = ty.PlaceholderType("T0")
    T1 = ty.PlaceholderType("T1")
    AppType = ty.CompositeType("App", parent=ty.GenericType, params=(T0, T1))
    AppType_Spec0 = AppType.clone_with(ty.Int, ty.Float)
    AppType_Spec1 = AppType.clone_with(ty.Int, ty.Float)
    assert AppType_Spec0 is AppType_Spec1


def test_PlaceholderType():
    T0 = ty.PlaceholderType("T0")
    T0_0 = ty.PlaceholderType("T0")

    assert T0 is not T0_0, "PlaceholderType should be unique"

    T0_1 = ty.GenericType("T0")
    assert T0 is not T0_1, "PlaceholderType should be unique"
    assert T0 == T0_0
    assert T0 != T0_1
