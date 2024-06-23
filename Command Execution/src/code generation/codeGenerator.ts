/**
 * A generic abstract class providing the necessary methods for code generation, like declaring variables, functions...etc
 * This class/interface needs to be implemented for each programming language that needs to be supported
 */

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
     * Declares a function in the target programming language.
     * @param name The name of the function.
     * @param returnType The return type of the function.
     * @param parameters The parameters of the function, each with a name and type.
     * @param body The body of the function as an array of strings, each representing a line of code.
     * @returns The code string for the function declaration.
     */
    abstract declareFunction(
        name: string,
        returnType: string,
        parameters: { name: string, type: string }[],
        body: string[]
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
     * Generates an import statement in the target programming language.
     * @param module The module to import from.
     * @param entities The entities to import from the module.
     * @returns The code string for the import statement.
     */
    abstract generateImport(module: string, entities: string[]): string;

    /**
     * Wraps the provided code lines into a code block, e.g., a function or class body.
     * @param lines The lines of code to wrap.
     * @returns The code string for the wrapped code block.
     */
    // abstract wrapInCodeBlock(lines: string[]): string;
}
