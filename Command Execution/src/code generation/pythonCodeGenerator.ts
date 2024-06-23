import { CodeGenerator } from "./codeGenerator";
import pythonReservedKeywords from "./language specifics/pythonReserved.json";
export class PythonCodeGenerator extends CodeGenerator {


    /**
     * Declare reserved keywords for each programming language
    **/
    protected reservedKeywords: Array<string>;


    // constructor
    constructor() {
        super();
        this.reservedKeywords = pythonReservedKeywords.reservedKeywords;
    }

    //**********************Utility functions**********************//
    /**
     * Check if the variable name is valid and not a reserved keyword
    **/
    private isValidVariableName(name: string): boolean {
        const pattern = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
        if (!pattern.test(name)) {
            return false;
        }
        return !this.reservedKeywords.includes(name);
    }

    private isValidConstantName(name: string): boolean {
        const pattern = /^[A-Z_][A-Z0-9_]*$/;
        if (!pattern.test(name)) {
            return false;
        }
        return !this.reservedKeywords.includes(name);
    }
        

    private isValidFunctionName(name: string): boolean {
        return this.isValidVariableName(name);
    }

    private isValidClassName(name: string): boolean {
        return this.isValidVariableName(name);
    }

    /**
     * wrap code in a code block with '`' character
    **/
    wrapInCodeBlock(lines: string[]): string {
        return lines.map(line => `    ${line}`).join('\n');
    }
    
    /**
     * Add Indentation to the code
     * 4 spaces before each line (if multiline)
    **/
    addIndentation(code: string): string {
        return code.split('\n').map(line => `    ${line}`).join('\n');
    }
    //********************************************//

    /**
     * Declare variables
    **/
    declareVariable(name: string, type?: string, initialValue?: any): string {
        if (!this.isValidVariableName(name)) {
            throw new Error(`Invalid variable name: ${name}`);
        }
        return `${name} = ${initialValue !== undefined ? initialValue : 'None'}`;
    }

    /**
     * Declare constants
    **/
    declareConstant(name: string, value: any): string {
        if (!this.isValidConstantName(name)) {
            throw new Error(`Invalid constant name: ${name}`);
        }
        return `${name} = ${value}`;
    }

    /**
     * Assign variables
    **/
    assignVariable(name: string, value: any): string {
        return `${name} = ${value}`;
    }

    /**
     * Declare function
     * Call function
     * Return statement
    **/
    declareFunction(name: string, returnType: string, parameters: { name: string, type: string }[], body: string[]): string {
        if (!this.isValidFunctionName(name)) {
            throw new Error(`Invalid function name: ${name}`);
        }
        const params = parameters.map(p => p.name).join(', ');
        const bodyCode = this.wrapInCodeBlock(body);

        return `def ${name}(${params}) -> ${returnType}:\n${bodyCode}`;
    }

    generateFunctionCall(name: string, args: string[]): string {
        return `${name}(${args.join(', ')})`;
    }

    generateReturn(value: string): string {
        return `return ${value}`;
    }

    /**
     * Declare Class
    **/
    declareClass(name: string, properties: { name: string, type: string }[], methods: string[]): string {
        if (!this.isValidClassName(name)) {
            throw new Error(`Invalid class name: ${name}`);
        }
        const props = properties.map(p => `self.${p.name} = None`).join('\n        ');
        const initMethod = `def __init__(self):\n        ${props}`;
        const methodCode = methods.join('\n\n    ');

        return `class ${name}:\n    ${initMethod}\n\n    ${methodCode}`;
    }

    /**
     * Import modules
    **/
    generateModuleImport(module: string, entities: string[]): string {
        return `from ${module} import ${entities.join(', ')}`;
    }

    generateImport(module: string): string {
        return `import ${module}`;
    }

    /**
     * Conditional statements
     * if, if-else
    **/
    generateIf(condition: string, body: string[]): string {
        return `if ${condition}:\n${this.wrapInCodeBlock(body)}`;
    }
    generateIfElse(condition: string, ifBody: string[], elseBody?: string[]): string {
        const ifCode = `if ${condition}:\n${this.wrapInCodeBlock(ifBody)}`;
        const elseCode = elseBody ? `\nelse:\n${this.wrapInCodeBlock(elseBody)}` : '';
        return `${ifCode}${elseCode}`;
    }

    /**
     * Loop statements
     * for, while, do-while
    **/
    generateForLoop(variable: string, iterable: string, body: string[]): string {
        const loopCode = `for ${variable} in ${iterable}:\n${this.wrapInCodeBlock(body)}`;
        return loopCode;
    }

    generateWhileLoop(condition: string, body: string[]): string {
        const loopCode = `while ${condition}:\n${this.wrapInCodeBlock(body)}`;
        return loopCode;
    }

    /**
     * Identity operators
     * is, is not
    **/
    generateIdentityOperation(left: string, operator: 'is' | 'is not', right: string): string {
        return `${left} ${operator} ${right}`;
    }

    /**
     * Membership operation 
     * in, not in
    **/
    generateMembershipOperation(left: string, operator: 'in' | 'not in', right: string): string {
        return `${left} ${operator} ${right}`;
    }

    /**
     * Logical operators
     * and, or, not
    **/
    generateLogicalOperation(left: string, operator: 'and' | 'or' | 'not', right: string): string {
        return `${left} ${operator} ${right}`;
    }

    /**
     * Comparison operators
     * <, >, <=, >=, ==, !=
    **/
    generateComparisonOperation(left: string, operator: '<' | '>' | '<=' | '>=' | '==' | '!=', right: string): string {
        return `${left} ${operator} ${right}`;
    }

    /**
     * Arithmetic operators
     * +, -, *, /, %, // , **
    **/
    generateArithmeticOperation(left: string, operator: '+' | '-' | '*' | '/' | '%' | '//' | '**', right: string): string {
        return `${left} ${operator} ${right}`;
    }

    /**
     * Bitwise operators
     * &, |, ^, ~, <<, >>
    **/
    generateBitwiseOperation(left: string, operator: '&' | '|' | '^' | '~' | '<<' | '>>', right: string): string {
        return `${left} ${operator} ${right}`;
    }



}
