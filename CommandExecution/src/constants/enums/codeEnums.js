"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CastingTypes = exports.ForLoop = exports.AssertionOperators = exports.UnaryOperators = exports.BitwiseOperators = exports.LogicalOperators = exports.ComparisonOperators = exports.ArithmeticOperators = exports.IdentityOperators = exports.MembershipOperators = exports.AssignmentOperators = exports.Whitespace = void 0;
var Whitespace;
(function (Whitespace) {
    Whitespace["Space"] = "space";
    Whitespace["Tab"] = "tab";
    Whitespace["NewLine"] = "newLine";
})(Whitespace || (exports.Whitespace = Whitespace = {}));
var AssignmentOperators;
(function (AssignmentOperators) {
    AssignmentOperators["Equals"] = "=";
    AssignmentOperators["PlusEquals"] = "+=";
    AssignmentOperators["MinusEquals"] = "-=";
    AssignmentOperators["MultiplyEquals"] = "*=";
    AssignmentOperators["DivideEquals"] = "/=";
    AssignmentOperators["FloorDivideEquals"] = "//=";
    AssignmentOperators["ModulusEquals"] = "%=";
    AssignmentOperators["ExponentEquals"] = "**=";
    AssignmentOperators["AndEquals"] = "&=";
    AssignmentOperators["OrEquals"] = "|=";
    AssignmentOperators["XorEquals"] = "^=";
    AssignmentOperators["LeftShiftEquals"] = "<<=";
    AssignmentOperators["RightShiftEquals"] = ">>=";
})(AssignmentOperators || (exports.AssignmentOperators = AssignmentOperators = {}));
var MembershipOperators;
(function (MembershipOperators) {
    MembershipOperators["In"] = "in";
    MembershipOperators["NotIn"] = "not in";
})(MembershipOperators || (exports.MembershipOperators = MembershipOperators = {}));
var IdentityOperators;
(function (IdentityOperators) {
    IdentityOperators["Is"] = "is";
    IdentityOperators["IsNot"] = "is not";
})(IdentityOperators || (exports.IdentityOperators = IdentityOperators = {}));
var ArithmeticOperators;
(function (ArithmeticOperators) {
    ArithmeticOperators["Addition"] = "+";
    ArithmeticOperators["Subtraction"] = "-";
    ArithmeticOperators["Multiplication"] = "*";
    ArithmeticOperators["Division"] = "/";
    ArithmeticOperators["FloorDivision"] = "//";
    ArithmeticOperators["Modulus"] = "%";
    ArithmeticOperators["Exponentiation"] = "**";
})(ArithmeticOperators || (exports.ArithmeticOperators = ArithmeticOperators = {}));
var ComparisonOperators;
(function (ComparisonOperators) {
    ComparisonOperators["Equal"] = "==";
    ComparisonOperators["NotEqual"] = "!=";
    ComparisonOperators["LessThan"] = "<";
    ComparisonOperators["GreaterThan"] = ">";
    ComparisonOperators["LessThanOrEqual"] = "<=";
    ComparisonOperators["GreaterThanOrEqual"] = ">=";
})(ComparisonOperators || (exports.ComparisonOperators = ComparisonOperators = {}));
var LogicalOperators;
(function (LogicalOperators) {
    LogicalOperators["And"] = "and";
    LogicalOperators["Or"] = "or";
    LogicalOperators["Not"] = "not";
})(LogicalOperators || (exports.LogicalOperators = LogicalOperators = {}));
var BitwiseOperators;
(function (BitwiseOperators) {
    BitwiseOperators["And"] = "&";
    BitwiseOperators["Or"] = "|";
    BitwiseOperators["Xor"] = "^";
    BitwiseOperators["LeftShift"] = "<<";
    BitwiseOperators["RightShift"] = ">>";
    BitwiseOperators["BitwiseNot"] = "~";
})(BitwiseOperators || (exports.BitwiseOperators = BitwiseOperators = {}));
var UnaryOperators;
(function (UnaryOperators) {
    UnaryOperators["Positive"] = "+";
    UnaryOperators["Negative"] = "-";
    UnaryOperators["Not"] = "not";
})(UnaryOperators || (exports.UnaryOperators = UnaryOperators = {}));
var AssertionOperators;
(function (AssertionOperators) {
    AssertionOperators["LessThanOrEqual"] = "<=";
    AssertionOperators["GreaterThanOrEqual"] = ">=";
    AssertionOperators["Equal"] = "==";
    AssertionOperators["NotEqual"] = "!=";
})(AssertionOperators || (exports.AssertionOperators = AssertionOperators = {}));
var ForLoop;
(function (ForLoop) {
    ForLoop["Range"] = "range";
    ForLoop["Iterable"] = "iterable";
    ForLoop["Enumerate"] = "enumerate";
})(ForLoop || (exports.ForLoop = ForLoop = {}));
var CastingTypes;
(function (CastingTypes) {
    CastingTypes["Integer"] = "int";
    CastingTypes["Float"] = "float";
    CastingTypes["String"] = "str";
    CastingTypes["Boolean"] = "bool";
    CastingTypes["List"] = "list";
    CastingTypes["Tuple"] = "tuple";
    CastingTypes["Set"] = "set";
    CastingTypes["Dictionary"] = "dict";
})(CastingTypes || (exports.CastingTypes = CastingTypes = {}));
//# sourceMappingURL=codeEnums.js.map