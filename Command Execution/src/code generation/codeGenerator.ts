/**
 * A generic abstract class providing the necessary methods for code generation, like declaring variables, functions...etc
 * This class/interface needs to be implemented for each programming language that needs to be supported
 */

import { ArithmeticOperators, BitwiseOperators, ComparisonOperators, IdentityOperators, LogicalOperators, MembershipOperators, Whitespace, AssertionOperators } from "../constants/enums/codeEnums";
import { AssignmentOperators } from "../constants/enums/codeEnums";

export abstract class CodeGenerator {

    /**
     * Declares a variable in the target programming language.
     * @param name The name of the variable.
     * @param type The type of the variable.
     * @param initialValue The initial value of the variable (optional).
     * @returns The code string for the variable declaration.
     */
    abstract declareVariable(name: string, type?: string, initialValue?: any): string;

    /**
     * Declares a constant in the target programming language.
     * @param name The name of the constant.
     * @param type The type of the constant.
     * @param value The value of the constant.
     * @returns The code string for the constant declaration.
     */
    abstract declareConstant(name: string, type?: string, value?: any): string;

    /**
     * Declares a function in the target programming language.
     * @param name The name of the function.
     * @param returnType The return type of the function.
     * @param parameters The parameters of the function, each with a name and type.
     * @param body The body of the function as an array of strings, each representing a line of code.
     * @returns The code string for the function declaration.
     */
    abstract declareFunction(
        name: string,
        parameters: { name: string, type: string }[],
        body?: string[],
        returnType?: string,
    ): string;

    /**
     * Declares a class in the target programming language.
     * @param name The name of the class.
     * @param properties The properties of the class, each with a name and type.
     * @param methods The methods of the class, each represented by a function declaration string.
     * @returns The code string for the class declaration.
     */
    abstract declareClass(
        name: string,
        properties: { name: string, type: string }[],
        methods: string[]
    ): string;

    /**
     * Assign a value to a variable in the target programming language.
     * @param name The name of the variable.
     * @param value The value to assign to the variable.
     * @returns The code string for the variable assignment.
     * @throws An error if the variable name is invalid.
     * @throws An error if the variable is not declared.
     * @throws An error if the value is not valid for the variable type.
     */
        abstract assignVariable(name: string, value: any, type: AssignmentOperators): string;


    /**
     * Generates an import statement in the target programming language.
     * @param module The module to import from.
     * @param entities The entities to import from the module.
     * @returns The code string for the import statement.
     */
    abstract generateImport(module: string, entities: string[]): string;
    abstract generateModuleImport(module: string, entities: string[]): string;

    /**
     * Generates a comment in the target programming language.
     * @param content The content of the comment.
     * @returns The code string for the comment.
     */
    abstract generateLineComment(content: string): string;
    abstract generateBlockComment(content: string): string;

    /**
     * Generates a function call in the target programming language.
     * @param name The name of the function to call.
     * @param args The arguments to pass to the function.
     * @returns The code string for the function call.
     */
    abstract generateFunctionCall(name: string, args: { name?: string, value: any }[]): string;


    /**
     * Generates a return statement in the target programming language.
     * @param value The value to return.
     * @returns The code string for the return statement.
     */
    abstract generateReturn(value?: string): string;

    /**
     * Generates a conditional statement in the target programming language.
     * @param condition The condition of the statement.
     * @param body The body of the statement as an array of strings, each representing a line of code.
     * @returns The code string for the conditional statement.
     * @throws An error if the condition is not valid.
    */
    abstract generateIf(condition: string, body: string[]): string;
    abstract generateIfElse(condition: string, body: string[]): string;

    /**
     * Generates a loop statement in the target programming language.
     * @param variable The variable to iterate over in the loop.
     * @param iterable The iterable to loop over.
     * @param body The body of the loop as an array of strings, each representing a line of code.
     * @returns The code string for the loop statement.
     */
    abstract generateForLoop(variable: string, iterable: string, body: string[]): string;
    abstract generateWhileLoop(condition: string, body: string[]): string;


    /**
     * Generates Identity Operator in the target programming language.
     * @param variable The variable to compare.
     * @param value The value to compare.
     * @returns The code string for the identity operator.
    */
    abstract generateIdentityOperation(left: string, operator: IdentityOperators, right: string): string;

    /**
     * Generates Membership Operator in the target programming language.
     * @param variable The variable to compare.
     * @param value The value to compare.
     * @returns The code string for the membership operator.
    */
    abstract generateMembershipOperation(left: string, operator: MembershipOperators, right: string): string;

    /**
     * Generates Logical Operator in the target programming language.
     * @param variable The variable to compare.
     * @param value The value to compare.
     * @returns The code string for the logical operator.
    */
    abstract generateLogicalOperation(left: string, operator: LogicalOperators, right: string): string;

    /**
     * Generates Comparison Operator in the target programming language.
     * @param variable The variable to compare.
     * @param value The value to compare.
     * @returns The code string for the comparison operator.
    */
    abstract generateComparisonOperation(left: string, operator: ComparisonOperators, right: string): string;

    /**
     * Generates Arithmetic Operator in the target programming language.
     * @param variable The variable to compare.
     * @param value The value to compare.
     * @returns The code string for the arithmetic operator.
    */
    abstract generateArithmeticOperation(left: string, operator: ArithmeticOperators, right: string): string;

    /**
     * Generates Bitwise Operator in the target programming language.
     * @param variable The variable to compare.
     * @param value The value to compare.
     * @returns The code string for the bitwise operator.
    */
    abstract generateBitwiseOperation(left: string, operator: BitwiseOperators, right: string): string;

    /**
     * Wraps the provided code lines into a code block, e.g., a function or class body.
     * @param lines The lines of code to wrap.
     * @returns The code string for the wrapped code block.
     */
    // abstract wrapInCodeBlock(lines: string[]): string;


    /**
     * Make Assertion in the target programming language.
     * @param variable The variable to compare.
     * @param value The value to compare.
     * @returns The code string for the assertion.
     * @throws An error if the assertion is not valid.
    */
    abstract generateAssertion(variable: string, value: any, type: AssertionOperators): string;

    /**
     * Add white spaces
     * 
     */
    abstract addWhiteSpace(type: Whitespace, count?: number): string;
}
