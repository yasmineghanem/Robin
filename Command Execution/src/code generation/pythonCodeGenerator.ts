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

    private isValidFunctionName(name: string): boolean {
        return this.isValidVariableName(name);
    }

    private isValidClassName(name: string): boolean {
        return this.isValidVariableName(name);
    }


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
     * Assign variables
    **/
    assignVariable(name: string, value: any): string {
        return `${name} = ${value}`;
    }

    /**
     * Declare function
    **/
    declareFunction(name: string, returnType: string, parameters: { name: string, type: string }[], body: string[]): string {
        if (!this.isValidFunctionName(name)) {
            throw new Error(`Invalid function name: ${name}`);
        }
        const params = parameters.map(p => p.name).join(', ');
        const bodyCode = this.wrapInCodeBlock(body);

        return `def ${name}(${params}) -> ${returnType}:\n${bodyCode}`;
    }

    /**
     * Declare Class
    **/
    declareClass(name: string, properties: { name: string, type: string }[], methods: string[]): string {
        const props = properties.map(p => `self.${p.name} = None`).join('\n        ');
        const initMethod = `def __init__(self):\n        ${props}`;
        const methodCode = methods.join('\n\n    ');

        return `class ${name}:\n    ${initMethod}\n\n    ${methodCode}`;
    }

    /**
     * Import modules
    **/
    generateSpecificImport(module: string, entities: string[]): string {
        return `from ${module} import ${entities.join(', ')}`;
    }

    generateImport(module: string): string {
        return `import ${module}`;
    }


    /**
     * wrap code in a code block with '`' character
    **/
    wrapInCodeBlock(lines: string[]): string {
        return lines.map(line => `    ${line}`).join('\n');
    }
}
