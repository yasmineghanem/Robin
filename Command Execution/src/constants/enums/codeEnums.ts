
export enum Whitespace {
    Space = 'space',
    Tab = 'tab',
    NewLine = 'newLine',
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