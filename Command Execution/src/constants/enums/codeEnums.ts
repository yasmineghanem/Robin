
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
    RightShift = '>>'
}

export enum UnaryOperators {
    Positive = '+',
    Negative = '-',
    Not = 'not',
    BitwiseNot = '~'
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

export enum ArithmeticOperator {
    Addition = "+",
    Subtraction = "-",
    Multiplication = "*",
    Division = "/",
    Modulus = "%",
    Exponentiation = "**",
    FloorDivision = "//",
}

export enum ComparisonOperator {
    EqualTo = "==",
    NotEqualTo = "!=",
    GreaterThan = ">",
    GreaterThanOrEqualTo = ">=",
    LessThan = "<",
    LessThanOrEqualTo = "<=",
}


export enum LogicalOperator {
    And = "and",
    Or = "or",
    Not = "not",
}

export enum BitwiseOperator {
    And = "&",
    Or = "|",
    Xor = "^",
    LeftShift = "<<",
    RightShift = ">>",
    Invert = "~",
}

export enum IdentityOperator {
    Is = "is",
    IsNot = "is not",

}

export enum MembershipOperator {
    In = "in",
    NotIn = "not in",
}
export type Operator = ArithmeticOperator | ComparisonOperator | LogicalOperator | BitwiseOperator;
