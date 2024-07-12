"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PythonCodeGenerator = void 0;
const codeEnums_1 = require("../constants/enums/codeEnums");
const codeGenerator_1 = require("./codeGenerator");
const pythonReserved_json_1 = __importDefault(require("./language specifics/pythonReserved.json"));
class PythonCodeGenerator extends codeGenerator_1.CodeGenerator {
    /**
     * Declare reserved keywords for each programming language
     **/
    tabString = "    ";
    editor;
    reservedKeywords;
    tabSize;
    // constructor
    constructor(editor) {
        super();
        this.reservedKeywords = pythonReserved_json_1.default.reservedKeywords;
        // TODO read tab size from .env
        this.tabSize = 4;
        this.editor = editor;
    }
    //**********************Utility functions**********************//
    /**
     * Check if the variable name is valid and not a reserved keyword
     **/
    isValidVariableName(name) {
        const pattern = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
        if (!pattern.test(name)) {
            return false;
        }
        return !this.reservedKeywords.includes(name);
    }
    isValidConstantName(name) {
        const pattern = /^[A-Z_][A-Z0-9_]*$/;
        if (!pattern.test(name)) {
            return false;
        }
        return !this.reservedKeywords.includes(name);
    }
    isValidFunctionName(name) {
        return this.isValidVariableName(name);
    }
    isValidClassName(name) {
        return this.isValidVariableName(name);
    }
    handleIndentationLevel(previous = false) {
        let currentLine;
        if (previous) {
            currentLine = this.editor.document.lineAt(Math.max(this.editor.selection.active.line - 1, 0)).text;
        }
        else {
            currentLine = this.editor.document.lineAt(this.editor.selection.active.line).text;
        }
        // find the number of white spaces in the beginning of the line,
        // and calculate the number of tabs
        let indentationLevel = 0;
        for (let i = 0; i < currentLine.length; i++) {
            if (currentLine[i] === " ") {
                indentationLevel++;
            }
            else {
                break;
            }
        }
        // check if first word in line is a scope
        currentLine = currentLine.trim();
        if (currentLine.endsWith(":")) {
            indentationLevel += this.tabSize;
        }
        // calculate the number of tabs
        const tabs = Math.floor(indentationLevel / this.tabSize);
        return tabs;
    }
    /**
     * wrap code in a code block with '`' character
     **/
    wrapInCodeBlock(lines) {
        return lines.map((line) => `    ${line}`).join("\n");
    }
    /**
     * Add Indentation to the code
     * 4 spaces before each line (if multiline)
     **/
    addIndentation(code) {
        // with tab_size
        return code
            .split("\n")
            .map((line) => `${this.addWhiteSpace(codeEnums_1.Whitespace.Tab, this.tabSize)}${line}`)
            .join("\n");
    }
    //********************************************//
    /**
     * Declare variables
     **/
    declareVariable(name, type, initialValue) {
        if (!this.isValidVariableName(name)) {
            throw new Error(`Invalid variable name: ${name}`);
        }
        const indentation = this.tabString.repeat(this.handleIndentationLevel(true)); // previous line
        if (type) {
            if (initialValue) {
                return `${name}: ${type} = ${initialValue}\n${indentation}`;
            }
            return `${name}: ${type}\n${indentation}`;
        }
        if (initialValue) {
            return `${name} = ${initialValue}\n${indentation}`;
        }
        return `${name}\n${indentation}`;
    }
    /**
     * Declare constants
     **/
    declareConstant(name, value) {
        if (!this.isValidConstantName(name.toUpperCase())) {
            throw new Error(`Invalid constant name: ${name}`);
        }
        return (`${name.toUpperCase()} = ${value}\n` + this.handleIndentationLevel(true));
    }
    /**
     * Assign variables
     **/
    assignVariable(name, value, type) {
        //Check before if RHS is same type as LHS
        ///////// we need function to check the type of the variable /////////
        if (!Object.values(codeEnums_1.AssignmentOperators).includes(type)) {
            throw new Error(`Invalid assignment type: ${type}`);
        }
        switch (type) {
            case codeEnums_1.AssignmentOperators.Equals:
                return `${name} = ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.PlusEquals:
                return `${name} += ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.MinusEquals:
                return `${name} -= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.MultiplyEquals:
                return `${name} *= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.DivideEquals:
                return `${name} /= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.FloorDivideEquals:
                return `${name} //= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.ModulusEquals:
                return `${name} %= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.ExponentEquals:
                return `${name} **= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.AndEquals:
                return `${name} &= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.OrEquals:
                return `${name} |= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.XorEquals:
                return `${name} ^= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.LeftShiftEquals:
                return `${name} <<= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case codeEnums_1.AssignmentOperators.RightShiftEquals:
                return `${name} >>= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            default:
                throw new Error(`Invalid assignment type: ${type}`);
        }
    }
    /**
     * Declare function
     * Call function
     * Return statement
     **/
    declareFunction(name, parameters, body, returnType) {
        if (!this.isValidFunctionName(name)) {
            throw new Error(`Invalid function name: ${name}`);
        }
        // check if valid parameters names
        if (parameters.some((p) => !this.isValidVariableName(p.name))) {
            throw new Error(`Invalid parameter name`);
        }
        // sort the parameters so that the ones without a value come first
        parameters.sort((a, b) => (a.value === undefined ? -1 : 1));
        const params = parameters
            .map((p) => `${p.name}${p.value
            ? ` = ${typeof p.value === "string" ? `"${p.value}"` : p.value}`
            : ""}`)
            .join(", ");
        const f = `def ${name}(${params}):\n` +
            this.tabString.repeat(this.handleIndentationLevel(true) + 1);
        return f;
    }
    generateFunctionCall(name, args) {
        const params = args
            .map((p) => (p.name ? `${p.name} = ${p.value}` : `${p.value}`))
            .join(", ");
        return (`${name}(${params}) \n` +
            this.tabString.repeat(this.handleIndentationLevel()));
    }
    generateReturn(value) {
        return (`return ${value ?? ""} ` +
            this.tabString.repeat(this.handleIndentationLevel(true)));
    }
    declareClass(name, properties, methods) {
        if (!this.isValidClassName(name)) {
            throw new Error(`Invalid class name: ${name} `);
        }
        // check if valid properties names
        if (properties.some((p) => !this.isValidVariableName(p.name))) {
            throw new Error(`Invalid property name`);
        }
        // check if valid method names
        if (methods.some((m) => !this.isValidFunctionName(m.name))) {
            throw new Error(`Invalid method name`);
        }
        let code = "";
        // add class name, capitalize first letter
        code += `class ${name.charAt(0).toUpperCase() + name.slice(1)}: \n`;
        // add constructor
        code += `${this.tabString}def __init__(self, ${properties
            .map((p) => p.name)
            .join(", ")}): \n`;
        // add properties
        properties.forEach((p) => {
            code += `${this.tabString}${this.tabString}self.${p.name} = ${p.name}\n`;
        });
        // add methods
        methods.forEach((m) => {
            // sort the parameters so that the one's without value come first
            m.parameters.sort((a, b) => (a.value === undefined ? -1 : 1));
            const params = m.parameters
                .map((p) => `${p.name}${p.value
                ? ` = ${typeof p.value === "string" ? `"${p.value}"` : p.value}`
                : ""}`)
                .join(", ");
            // code += "\n";
            code += `\n${this.tabString}def ${m.name}(self, ${params}):\n${this.tabString}`;
            code += this.wrapInCodeBlock(m.body ?? ["pass\n"]);
        });
        return code;
    }
    /**
     * Import modules
     **/
    generateImportModule(library, modules) {
        return `from ${library} import ${modules.join(", ")}\n`;
    }
    generateImportLibrary(library) {
        return `import ${library}\n`;
    }
    /**
     * Conditional statements
     * if, if-else
     **/
    generateIf(condition, body) {
        return `if ${condition}: \n${this.wrapInCodeBlock(body)} `;
    }
    generateIfElse(condition, ifBody, elseBody) {
        const ifCode = `if ${condition}: \n${this.wrapInCodeBlock(ifBody)} `;
        const elseCode = elseBody
            ? `\nelse: \n${this.wrapInCodeBlock(elseBody)} `
            : "";
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
    generateForLoop(type, params, body) {
        switch (type) {
            case codeEnums_1.ForLoop.Range:
                return this.generateRangeLoop(params);
            case codeEnums_1.ForLoop.Iterable:
                return this.generateIterableLoop(params);
            case codeEnums_1.ForLoop.Enumerate:
                return this.generateEnumerateLoop(params);
            default:
                return `for `;
        }
    }
    generateIterableLoop(params, body) {
        const loopCode = `for ${params.iterators.join(", ")} in ${params.iterable}: \n${this.wrapInCodeBlock(body ?? [""])} `;
        return loopCode;
    }
    generateRangeLoop(params
    // body?: string[]
    ) {
        const { iterators, start, end, step } = params;
        let forLoop = "";
        let actualStart = start ?? 0;
        let actualEnd = end ?? 0;
        let actualStep = step ?? 1;
        if (actualStart < actualEnd) {
            forLoop = `for ${iterators.join(", ")} in range(${actualStart}, ${actualEnd}, ${actualStep}): \n`;
        }
        else {
            forLoop = `for ${iterators.join(", ")} in range(${actualStart}, ${actualEnd}, ${-actualStep}): \n`;
        }
        return forLoop + this.tabString.repeat(this.handleIndentationLevel() + 1);
    }
    generateEnumerateLoop(params, body) {
        const { iterators, iterable, start } = params;
        return (`for ${iterators.join(", ")} in enumerate(${iterable}${start ? ` ,${start}` : ""}): \n` + this.tabString.repeat(this.handleIndentationLevel(true) + 1));
    }
    generateWhileLoop(condition, body) {
        const conditionCode = condition
            .map((c) => `${c.logicalOperator ?? ""} ${c.left} ${c.operator} ${c.right}`)
            .join(" ");
        const loopCode = `while ${conditionCode}: \n${this.wrapInCodeBlock(body ?? [""])} `;
        return loopCode;
    }
    /**
     * Try Except
     */
    generateTryExcept(tryBody, exception, exceptionInstance, exceptBody) {
        const tryCode = `try: \n${this.wrapInCodeBlock(tryBody ?? [""])} `;
        const exceptCode = `except ${exception} as ${exceptionInstance}: \n${this.wrapInCodeBlock(exceptBody ?? [""])} `;
        return `${tryCode} \n${exceptCode} `;
    }
    // /**
    //  * Identity operators
    //  * is, is not
    //  **/
    // generateIdentityOperation(
    //   left: string,
    //   operator: IdentityOperators,
    //   right: string
    // ): string {
    //   return `${left} ${operator} ${right} `;
    // }
    // /**
    //  * Membership operation
    //  * in, not in
    //  **/
    // generateMembershipOperation(
    //   left: string,
    //   operator: MembershipOperators,
    //   right: string
    // ): string {
    //   return `${left} ${operator} ${right} `;
    // }
    // /**
    //  * Logical operators
    //  * and, or, not
    //  **/
    // generateLogicalOperation(
    //   left: string,
    //   operator: LogicalOperators,
    //   right: string
    // ): string {
    //   return `${left} ${operator} ${right} `;
    // }
    // /**
    //  * Comparison operators
    //  * <, >, <=, >=, ==, !=
    //  **/
    // generateComparisonOperation(
    //   left: string,
    //   operator: ComparisonOperators,
    //   right: string
    // ): string {
    //   return `${left} ${operator} ${right} `;
    // }
    // /**
    //  * Arithmetic operators
    //  * +, -, *, /, %, // , **
    //  **/
    // generateArithmeticOperation(
    //   left: string,
    //   operator: ArithmeticOperators,
    //   right: string
    // ): string {
    //   return `${left} ${operator} ${right} `;
    // }
    // /**
    //  * Bitwise operators
    //  * &, |, ^, ~, <<, >>
    //  **/
    // generateBitwiseOperation(
    //   left: string,
    //   operator: BitwiseOperators,
    //   right: string
    // ): string {
    //   return `${left} ${operator} ${right} `;
    // }
    /**
     * Assertion
     **/
    generateAssertion(variable, value, type) {
        if (!Object.values(codeEnums_1.AssertionOperators).includes(type)) {
            throw new Error(`Invalid assertion type: ${type}`);
        }
        switch (type) {
            case codeEnums_1.AssertionOperators.Equal:
                return (`assert ${variable} == ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.AssertionOperators.NotEqual:
                return (`assert ${variable} != ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.AssertionOperators.GreaterThanOrEqual:
                return (`assert ${variable} >= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.AssertionOperators.LessThanOrEqual:
                return (`assert ${variable} <= ${value}\n` + this.tabString.repeat(this.handleIndentationLevel(true)));
            default:
                throw new Error(`Invalid assertion type: ${type}`);
        }
    }
    /**
     * Generate Casting
     **/
    generateCasting(variable, type) {
        if (!Object.values(codeEnums_1.CastingTypes).includes(type)) {
            throw new Error(`Invalid casting type: ${type}`);
        }
        switch (type) {
            case codeEnums_1.CastingTypes.Integer:
                return (`${variable} = int(${variable})\n` + this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.CastingTypes.Float:
                return (`${variable} = float(${variable})\n` +
                    this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.CastingTypes.String:
                return (`${variable} = str(${variable})\n` + this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.CastingTypes.Boolean:
                return (`${variable} = bool(${variable})\n` +
                    this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.CastingTypes.List:
                return (`${variable} = list(${variable})\n` +
                    this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.CastingTypes.Tuple:
                return (`${variable} = tuple(${variable})\n` +
                    this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.CastingTypes.Set:
                return (`${variable} = set(${variable})\n` + this.tabString.repeat(this.handleIndentationLevel(true)));
            case codeEnums_1.CastingTypes.Dictionary:
                return (`${variable} = dict(${variable})\n` +
                    this.tabString.repeat(this.handleIndentationLevel(true)));
            default:
                throw new Error(`Invalid casting type: ${type}`);
        }
    }
    /**
     * Generate User Input
     **/
    generateUserInput(variable, message) {
        return (`${variable} = input('${message ? message : ""}')\n` +
            this.tabString.repeat(this.handleIndentationLevel(true)));
    }
    /**
     * Generate Print
     **/
    generatePrint(value, type) {
        switch (type) {
            case "string":
                return `print("${value}")\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            case "variable":
                return `print(${value})\n` + this.tabString.repeat(this.handleIndentationLevel(true));
            default:
                return `print(${value})\n` + this.tabString.repeat(this.handleIndentationLevel(true));
        }
    }
    /**
     * Read file
     **/
    //TODO: add options to read line, read all file, read character
    generateReadFile(path, variable) {
        return (`${variable} = open("${path}", 'r').read()\n` +
            this.tabString.repeat(this.handleIndentationLevel(true)));
    }
    /**
     * Write file
     **/
    generateWriteFile(path, content) {
        return (`open("${path}", 'w').write("${content}")\n` +
            this.tabString.repeat(this.handleIndentationLevel(true)));
    }
    /**
     * White spaces
     */
    addWhiteSpace(type, count) {
        let ws;
        switch (type) {
            case codeEnums_1.Whitespace.Space:
                ws = " ";
                break;
            case codeEnums_1.Whitespace.Tab:
                ws = "    ";
                break;
            case codeEnums_1.Whitespace.NewLine:
                ws = "\n";
                break;
            default:
                ws = " ";
        }
        return ws.repeat(count ?? 1);
    }
    /**
     * Comments
     * Single line comments
     * Multi line comments
     **/
    generateLineComment(content) {
        return `# ${content}\n` + this.tabString.repeat(this.handleIndentationLevel(true));
    }
    generateBlockComment(content) {
        return (`''' ${content.join("\n")} '''\n` + this.tabString.repeat(this.handleIndentationLevel(true)));
    }
    generateOperation(left, operator, right) {
        return (`${left} ${operator} ${right} \n` + this.handleIndentationLevel(true));
    }
    generateConditional(conditions) {
        let code = "";
        conditions.forEach((c) => {
            if (c.keyword === "if" || c.keyword === "elif") {
                code += `${c.keyword} ${c.condition
                    ?.map((cond) => `${cond.logicalOperator ?? ""} ${cond.left} ${cond.operator} ${cond.right}`)
                    .join(" ")}: \n`;
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
    generateArrayOperation(name, operation) {
        // pack, uno
        return "";
    }
    exitScope() {
        let indentationLevel = this.handleIndentationLevel();
        return `\n${this.addWhiteSpace(codeEnums_1.Whitespace.Tab, Math.max(indentationLevel - 1, 0))}`.repeat(2);
    }
}
exports.PythonCodeGenerator = PythonCodeGenerator;
//# sourceMappingURL=pythonCodeGenerator.js.map