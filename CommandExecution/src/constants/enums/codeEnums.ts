import exp from "constants";
import e from "express";

export enum Whitespace {
    Space = 'space',
    Tab = 'tab',
    NewLine = 'newLine',
}

export enum AssignmentOperators {
    Equals = '=',
    PlusEquals = '+=',
    MinusEquals = '-=',
    MultiplyEquals = '*=',
    DivideEquals = '/=',
    FloorDivideEquals = '//=',
    ModulusEquals = '%=',
    ExponentEquals = '**=',
    AndEquals = '&=',
    OrEquals = '|=',
    XorEquals = '^=',
    LeftShiftEquals = '<<=',
    RightShiftEquals = '>>='
}

export enum MembershipOperators {
    In = 'in',
    NotIn = 'not in'
}

export enum IdentityOperators {
    Is = 'is',
    IsNot = 'is not'
}

export enum ArithmeticOperators {
    Addition = '+',
    Subtraction = '-',
    Multiplication = '*',
    Division = '/',
    FloorDivision = '//',
    Modulus = '%',
    Exponentiation = '**'
}

export enum ComparisonOperators {
    Equal = '==',
    NotEqual = '!=',
    LessThan = '<',
    GreaterThan = '>',
    LessThanOrEqual = '<=',
    GreaterThanOrEqual = '>='
}

export enum LogicalOperators {
    And = 'and',
    Or = 'or',
    Not = 'not'
}

export enum BitwiseOperators {
    And = '&',
    Or = '|',
    Xor = '^',
    LeftShift = '<<',
    RightShift = '>>',
    BitwiseNot = '~'

}

export enum UnaryOperators {
    Positive = '+',
    Negative = '-',
    Not = 'not',
}

export enum AssertionOperators {
    LessThanOrEqual = '<=',
    GreaterThanOrEqual = '>=',
    Equal = '==',
    NotEqual = '!='
}
export enum ForLoop {
    Range = "range",
    Iterable = "iterable",
    Enumerate = "enumerate",
}

export enum CastingTypes {
    Integer = 'int',
    Float = 'float',
    String = 'str',
    Boolean = 'bool',
    List = 'list',
    Tuple = 'tuple',
    Set = 'set',
    Dictionary = 'dict'
}


export type Operator = ArithmeticOperators | ComparisonOperators | LogicalOperators | BitwiseOperators | MembershipOperators;
