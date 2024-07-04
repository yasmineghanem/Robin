import { ArithmeticOperators, AssertionOperators, AssignmentOperators, BitwiseOperators, CastingTypes, ComparisonOperators, ForLoop, IdentityOperators, LogicalOperators, MembershipOperators, Operator, Whitespace } from "../constants/enums/codeEnums";
import { CodeGenerator } from "./codeGenerator";
import pythonReservedKeywords from "./language specifics/pythonReserved.json";


interface Condition {
    logicalOperator?: LogicalOperators,
    left: string;
    operator: Operator;
    right: string;

}
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
        return `${name} = ${initialValue !== undefined ? initialValue : 'None'}\n`;
    }

    /**
     * Declare constants
    **/
    declareConstant(name: string, value: any): string {
        if (!this.isValidConstantName(name.toUpperCase())) {
            throw new Error(`Invalid constant name: ${name}`);
        }
        return `${name.toUpperCase()} = ${value}\n`;
    }

    /**
     * Assign variables
    **/
    assignVariable(name: string, value: any, type: string): string {
        //Check before if RHS is same type as LHS
        ///////// we need function to check the type of the variable /////////
        if (!Object.values(AssignmentOperators).includes(type as AssignmentOperators)) {
            throw new Error(`Invalid assignment type: ${type}`);
        }
        switch (type) {
            case AssignmentOperators.Equals:
                return `${name} = ${value}\n`;
            case AssignmentOperators.PlusEquals:
                return `${name} += ${value}\n`;
            case AssignmentOperators.MinusEquals:
                return `${name} -= ${value}\n`;
            case AssignmentOperators.MultiplyEquals:
                return `${name} *= ${value}\n`;
            case AssignmentOperators.DivideEquals:
                return `${name} /= ${value}\n`;
            case AssignmentOperators.FloorDivideEquals:
                return `${name} //= ${value}\n`;
            case AssignmentOperators.ModulusEquals:
                return `${name} %= ${value}\n`;
            case AssignmentOperators.ExponentEquals:
                return `${name} **= ${value}\n`;
            case AssignmentOperators.AndEquals:
                return `${name} &= ${value}\n`;
            case AssignmentOperators.OrEquals:
                return `${name} |= ${value}\n`;
            case AssignmentOperators.XorEquals:
                return `${name} ^= ${value}\n`;
            case AssignmentOperators.LeftShiftEquals:
                return `${name} <<= ${value}\n`;
            case AssignmentOperators.RightShiftEquals:
                return `${name} >>= ${value}\n`;
            default:
                throw new Error(`Invalid assignment type: ${type}`);
        }
    }

    /**
     * Declare function
     * Call function
     * Return statement
    **/
    declareFunction(name: string, parameters: { name: string, value?: any }[], body?: string[], returnType?: string,): string {
        if (!this.isValidFunctionName(name)) {
            throw new Error(`Invalid function name: ${name}`);
        }

        // check if valid parameters names
        if (parameters.some(p => !this.isValidVariableName(p.name))) {
            throw new Error(`Invalid parameter name`);
        }

        // sort the parameters so that the ones without a value come first
        parameters.sort((a, b) => a.value === undefined ? -1 : 1);

        const params = parameters.map(p => `${p.name}${p.value ? ` = ${typeof p.value === "string" ? `"${p.value}"` : p.value}` : ''}`).join(', ');
        const functionHeader = `def ${name}(${params}):`;
        const functionBody = this.wrapInCodeBlock(body ?? [""]);
        return `${functionHeader}\n${functionBody}`;
    }

    // {
    //     "name": "x_string",
    //         "args": [
    //             {
    //                 "name": "x",
    //                 "value": "test"
    //             },
    //             {
    //                 "name": "y",
    //                 "value": "test"
    //             }
    //         ]
    // }

    generateFunctionCall(name: string, args: { name?: string, value: any }[]): string {
        const params = args.map(p =>
            p.name ? `${p.name} = 
                ${typeof p.value === "string" ? `"${p.value}"` : p.value}` :
                typeof p.value === "string" ? `"${p.value}"` : p.value
        ).join(', ');
        return `${name}(${params}) \n`;

    }

    generateReturn(value?: string): string {
        return `return ${value ?? ""} `;
    }

    /**
     * Declare Class
     * class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

        def myfunc(self):
            print("Hello my name is " + self.name)

    **/
    declareClass(name: string, properties: { name: string, type?: string }[], methods: { name: string, parameters: { name: string, value?: any }[], body?: string[], }[]): string {
        if (!this.isValidClassName(name)) {
            throw new Error(`Invalid class name: ${name} `);
        }

        // check if valid properties names
        if (properties.some(p => !this.isValidVariableName(p.name))) {
            throw new Error(`Invalid property name`);
        }

        // check if valid method names
        if (methods.some(m => !this.isValidFunctionName(m.name))) {
            throw new Error(`Invalid method name`);
        }

        let code = "";
        // add class name, capitalize first letter
        code += `class ${name.charAt(0).toUpperCase() + name.slice(1)}: \n`;

        // add constructor
        code += `\tdef __init__(self, ${properties.map(p => p.name).join(', ')}): \n`;

        // add properties
        properties.forEach(p => {
            code += `\t\tself.${p.name} = ${p.name}\n`;
        });

        // add methods
        methods.forEach(m => {
            // sort the parameters so that the one's without value come first
            m.parameters.sort((a, b) => a.value === undefined ? -1 : 1);
            const params = m.parameters.map(p => `${p.name}${p.value ? ` = ${typeof p.value === "string" ? `"${p.value}"` : p.value}` : ''}`).join(', ');
            // code += "\n";
            code += `\n\tdef ${m.name}(self, ${params}):\n\t`;
            code += this.wrapInCodeBlock(m.body ?? ['pass\n']);
        });

        return code;

    }

    /**
     * Import modules
    **/
    generateImportModule(library: string, modules: string[]): string {
        return `from ${library} import ${modules.join(', ')}\n`;
    }

    generateImportLibrary(library: string): string {
        return `import ${library}\n`;
    }

    /**
     * Conditional statements
     * if, if-else
    **/
    generateIf(condition: string, body: string[]): string {
        return `if ${condition}: \n${this.wrapInCodeBlock(body)} `;
    }
    generateIfElse(condition: string, ifBody: string[], elseBody?: string[]): string {
        const ifCode = `if ${condition}: \n${this.wrapInCodeBlock(ifBody)} `;
        const elseCode = elseBody ? `\nelse: \n${this.wrapInCodeBlock(elseBody)} ` : '';
        return `${ifCode}${elseCode} `;
    }

    /**
     * Loop statements
     * for, while, do-while
    **/
    // generateForLoop(variable: string, iterable: string, body: string[]): string {
    //     const loopCode = `for ${variable} in ${iterable}: \n${this.wrapInCodeBlock(body)} `;
    //     return loopCode;
    // }
    generateForLoop(type: ForLoop, params: any, body?: string[]): string {
        switch (type) {
            case ForLoop.Range:
                return this.generateRangeLoop(params, body);
            case ForLoop.Iterable:
                return this.generateIterableLoop(params, body);
            case ForLoop.Enumerate:
                return this.generateEnumerateLoop(params, body);
            default:
                return `for `;
        }

    }


    generateIterableLoop(
        params: { iterators: string[], iterable: string },
        body?: string[]
    ): string {
        const loopCode = `for ${params.iterators.join(", ")} in ${params.iterable}: \n${this.wrapInCodeBlock(body ?? [''])} `;
        return loopCode;
    }



    generateRangeLoop(
        params: { iterators: string[], start: number, end: number, step?: number },
        body?: string[]
    ): string {
        const { iterators, start, end, step } = params;
        // if there's no step and no start, just return range of the end
        if (!step && !start) {
            return `for ${iterators.join(", ")} in range(${end ?? ''}): \n${this.wrapInCodeBlock(body ?? [''])} `;
        }
        // if there's no step, just return range of start and end
        if (!step) {
            return `for ${iterators.join(", ")} in range(${start ? start + ", " : ""} ${end ?? ''}): \n${this.wrapInCodeBlock(body ?? [''])} `;
        }
        // if there's a step, return range of start, end and step
        return `for ${iterators.join(", ")} in range(${start ? start : "0"} ,${end ?? ''}, ${step}): \n${this.wrapInCodeBlock(body ?? [''])} `;
    }


    generateEnumerateLoop(params: { iterators: string[], iterable: string, start?: number }, body?: string[]): string {
        const { iterators, iterable, start } = params;
        return `for ${iterators.join(", ")} in enumerate(${iterable}${start ? ` ,${start}` : ''}): \n${this.wrapInCodeBlock(body ?? [''])} `;

    }


    generateWhileLoop(condition: Condition[], body?: string[]): string {

        const conditionCode = condition.map(c => `${c.logicalOperator ?? ""} ${c.left} ${c.operator} ${c.right}`).join(' ');
        const loopCode = `while ${conditionCode}: \n${this.wrapInCodeBlock(body ?? [''])} `;
        return loopCode;
    }

    /**
     * Identity operators
     * is, is not
    **/
    generateIdentityOperation(left: string, operator: IdentityOperators, right: string): string {
        return `${left} ${operator} ${right} `;
    }

    /**
     * Membership operation 
     * in, not in
    **/
    generateMembershipOperation(left: string, operator: MembershipOperators, right: string): string {
        return `${left} ${operator} ${right} `;
    }

    /**
     * Logical operators
     * and, or, not
    **/
    generateLogicalOperation(left: string, operator: LogicalOperators, right: string): string {
        return `${left} ${operator} ${right} `;
    }

    /**
     * Comparison operators
     * <, >, <=, >=, ==, !=
    **/
    generateComparisonOperation(left: string, operator: ComparisonOperators, right: string): string {
        return `${left} ${operator} ${right} `;
    }

    /**
     * Arithmetic operators
     * +, -, *, /, %, // , **
    **/
    generateArithmeticOperation(left: string, operator: ArithmeticOperators, right: string): string {
        return `${left} ${operator} ${right} `;
    }

    /**
     * Bitwise operators
     * &, |, ^, ~, <<, >>
    **/
    generateBitwiseOperation(left: string, operator: BitwiseOperators, right: string): string {
        return `${left} ${operator} ${right} `;
    }

    /**
     * Assertion
    **/
    generateAssertion(variable: string, value: any, type: string): string {
        if (!Object.values(AssertionOperators).includes(type as AssertionOperators)) {
            throw new Error(`Invalid assertion type: ${type}`);
        }
        switch (type) {
            case AssertionOperators.Equal:
                return `assert ${variable} == ${value}\n`;
            case AssertionOperators.NotEqual:
                return `assert ${variable} != ${value}\n`;
            case AssertionOperators.GreaterThanOrEqual:
                return `assert ${variable} >= ${value}\n`;
            case AssertionOperators.LessThanOrEqual:
                return `assert ${variable} <= ${value}\n`;
            default:
                throw new Error(`Invalid assertion type: ${type}`);
        }

    }

    /**
     * Generate Casting
    **/
    generateCasting(value: any, type: string): string {
        if (!Object.values(CastingTypes).includes(type as CastingTypes)) {
            throw new Error(`Invalid casting type: ${type}`);
        }
        switch (type) {
            case CastingTypes.Integer:
                return `int(${value})\n`;
            case CastingTypes.Float:
                return `float(${value})\n`;
            case CastingTypes.String:
                return `str(${value})\n`;
            case CastingTypes.Boolean:
                return `bool(${value})\n`;
            case CastingTypes.List:
                return `list(${value})\n`;
            case CastingTypes.Tuple:
                return `tuple(${value})\n`;
            case CastingTypes.Set:
                return `set(${value})\n`;
            case CastingTypes.Dictionary:
                return `dict(${value})\n`;
            default:
                throw new Error(`Invalid casting type: ${type}`);
        }
    }

    /**
     * Generate User Input
    **/
    generateUserInput(variable: string, message?: string | undefined): string {
        return `${variable} = input(${message ? message : ''})\n`;
    }

    /**
     * Generate Print
    **/
    generatePrint(value: any, type: string): string {
        switch (type) {
            case 'string':
                return `print("${value}")\n`;
            case 'variable':
                return `print(${value})\n`;
            default:
                return `print w khalas(${value})\n`;
        }
    }

    /**
     * Read file
    **/
    //TODO: add options to read line, read all file, read character
    generateReadFile(path: string, variable: any): string {
        return `${variable} = open("${path}", 'r').read()\n`;
    }

    /**
     * Write file
    **/
    generateWriteFile(path: string, content: any): string {
        return `open("${path}", 'w').write("${content}")\n`;
    }

    /**
     * White spaces
     */
    addWhiteSpace(
        type: Whitespace,
        count?: number,
    ): string {
        let ws;
        switch (type) {
            case Whitespace.Space:
                ws = ' ';
                break;
            case Whitespace.Tab:
                ws = '\t';
                break;
            case Whitespace.NewLine:
                ws = '\n';
                break;
            default:
                ws = ' ';

        }
        return ws.repeat(count ?? 1);

    }

    /**
     * Comments
     * Single line comments
     * Multi line comments
    **/
    generateLineComment(content: string): string {
        return `# ${content}\n`;
    }

    generateBlockComment(content: string[]): string {
        return `''' ${content.join("\n")} '''\n`;
    }

    generateOperation(
        left: string,
        operator: Operator,
        right: string
    ): string {
        return `${left} ${operator} ${right} `;
    }

    // [
    //     {
    //         "keyword": "if",
    //         "condition": [
    //             {
    //                 "left": "x",
    //                 "operator": ">",
    //                 "right": "5"
    //             },
    //             {
    //                 "logicalOperator": "and",
    //                 "left": "x",
    //                 "operator": ">",
    //                 "right": "5"
    //             }
    //         ]
    //     },
    //     {
    //         "keyword": "else"
    //     }
    // ]
    generateConditional(
        conditions: { keyword: 'if' | 'else' | 'elif', condition?: Condition[], body?: string[] }[]
    ): string {
        let code = '';
        conditions.forEach(c => {
            if (c.keyword === 'if' || c.keyword === 'elif') {
                code += `${c.keyword} ${c.condition?.map(cond => `${cond.logicalOperator ?? ""} ${cond.left} ${cond.operator} ${cond.right}`).join(' ')}: \n`;
            }

            else {
                code += `\nelse: \n`;
                // body
                // if (c.body) {
                //     code += `${this.wrapInCodeBlock(c.body)} `;
                // }
            }
        });
        return code;
    }

    generateArrayOperation(
        name: string,
        operation: string,
    ): string {
        // pack, uno
        return ""
    }
}
