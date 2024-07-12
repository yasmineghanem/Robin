import vscode from "vscode";
import {
  AssertionOperators,
  AssignmentOperators,
  CastingTypes,
  ForLoop,
  LogicalOperators,
  Operator,
  Whitespace,
} from "../constants/enums/codeEnums";
import { CodeGenerator } from "./codeGenerator";
import pythonSpecifics from "./language specifics/pythonReserved.json";

interface Condition {
  logicalOperator?: LogicalOperators;
  left: string;
  operator: Operator;
  right: string;
}
export class PythonCodeGenerator extends CodeGenerator {
  /**
   * Declare reserved keywords for each programming language
   **/
  private tabString: string = "    ";
  private editor: vscode.TextEditor;
  protected typeMappings: { [key: string]: string };
  protected reservedKeywords: Array<string>;
  protected tabSize: number;

  // constructor
  constructor(editor: vscode.TextEditor) {
    super();
    this.reservedKeywords = pythonSpecifics.reservedKeywords;
    this.typeMappings = pythonSpecifics.typeMappings;
    // TODO read tab size from .env
    this.tabSize = 4;
    this.editor = editor;
  }

  //**********************Utility functions**********************//
  /**
   * Check if the variable name is valid and not a reserved keyword
   **/
  private isValidVariableName(name: string): boolean {
    // swap spaces with underscores
    name = name.replace(/ /g, "_");

    const pattern = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
    if (!pattern.test(name)) {
      return false;
    }
    return !this.reservedKeywords.includes(name);
  }

  private isValidConstantName(name: string): boolean {
    // swap spaces with underscores
    name = name.replace(/ /g, "_");
    const pattern = /^[A-Z_][A-Z0-9_]*$/;
    if (!pattern.test(name)) {
      return false;
    }
    return !this.reservedKeywords.includes(name);
  }

  private handleIndentationLevel(previous: boolean = true): number {
    let currentLine;
    if (previous) {
      currentLine = this.editor.document.lineAt(
        Math.max(this.editor.selection.active.line - 1, 0)
      ).text;
    } else {
      currentLine = this.editor.document.lineAt(
        this.editor.selection.active.line
      ).text;
    }
    // find the number of white spaces in the beginning of the line,
    // and calculate the number of tabs
    let indentationLevel = 0;
    for (let i = 0; i < currentLine.length; i++) {
      if (currentLine[i] === " ") {
        indentationLevel++;
      } else {
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

  // check if the cursor position is in the same level as the indentation level
  private checkCursorPosition(indentationLevel: number): any {
    // let currentLine = this.editor.document.lineAt(
    //   this.editor.selection.active.line
    // ).text;
    let currentIndentationLevel = this.handleIndentationLevel(false);
    if (currentIndentationLevel === indentationLevel) {
      return true;
    }
    return false;
  }

  /**
   * wrap code in a code block with '`' character
   **/
  wrapInCodeBlock(lines: string[]): string {
    return lines.map((line) => `    ${line}`).join("\n");
  }

  /**
   * Add Indentation to the code
   * 4 spaces before each line (if multiline)
   **/
  addIndentation(code: string): string {
    // with tab_size
    return code
      .split("\n")
      .map(
        (line) => `${this.addWhiteSpace(Whitespace.Tab, this.tabSize)}${line}`
      )
      .join("\n");
  }
  //********************************************//
  /**
   * Declare variables
   **/
  declareVariable(name: string, type?: string, initialValue?: any): string {
    if (!this.isValidVariableName(name)) {
      throw new Error(`Invalid variable name: ${name}`);
    }

    const indentation = this.tabString.repeat(
      this.handleIndentationLevel(true)
    ); // previous line
    if (type) {
      let mappedType: string =
        this.typeMappings[type.toLowerCase() as keyof typeof this.typeMappings];

      if (initialValue) {
        return `${name}: ${mappedType} = ${initialValue}\n${indentation}`;
      }

      return `${name}: ${mappedType}\n${indentation}`;
    }

    if (initialValue) {
      return `${name} = ${initialValue}\n${indentation}`;
    }

    return `${name}\n${indentation}`;
  }

  /**
   * Declare constants
   **/
  declareConstant(name: string, value: any): string {
    if (!this.isValidConstantName(name.toUpperCase())) {
      throw new Error(`Invalid constant name: ${name}`);
    }
    return (
      `${name.toUpperCase()} = ${value}\n` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
  }

  /**
   * Assign variables
   **/
  assignVariable(name: string, value: any, type: string): string {
    //Check before if RHS is same type as LHS
    ///////// we need function to check the type of the variable /////////
    if (
      !Object.values(AssignmentOperators).includes(type as AssignmentOperators)
    ) {
      throw new Error(`Invalid assignment type: ${type}`);
    }
    switch (type) {
      case AssignmentOperators.Equals:
        return (
          `${name} = ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.PlusEquals:
        return (
          `${name} += ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.MinusEquals:
        return (
          `${name} -= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.MultiplyEquals:
        return (
          `${name} *= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.DivideEquals:
        return (
          `${name} /= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.FloorDivideEquals:
        return (
          `${name} //= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.ModulusEquals:
        return (
          `${name} %= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.ExponentEquals:
        return (
          `${name} **= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.AndEquals:
        return (
          `${name} &= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.OrEquals:
        return (
          `${name} |= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.XorEquals:
        return (
          `${name} ^= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.LeftShiftEquals:
        return (
          `${name} <<= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssignmentOperators.RightShiftEquals:
        return (
          `${name} >>= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      default:
        throw new Error(`Invalid assignment type: ${type}`);
    }
  }

  /**
   * Declare function
   * Call function
   * Return statement
   **/
  declareFunction(
    name: string,
    parameters: { name: string; value?: any }[],
    body?: string[],
    returnType?: string
  ): string {
    if (!this.isValidVariableName(name)) {
      throw new Error(`Invalid function name: ${name}`);
    }

    // check if valid parameters names
    if (parameters.some((p) => !this.isValidVariableName(p.name))) {
      throw new Error(`Invalid parameter name`);
    }

    // sort the parameters so that the ones without a value come first
    parameters.sort((a, b) => (a.value === undefined ? -1 : 1));

    const params = parameters
      .map(
        (p) =>
          `${p.name}${
            p.value
              ? ` = ${typeof p.value === "string" ? `"${p.value}"` : p.value}`
              : ""
          }`
      )
      .join(", ");
    const f =
      `def ${name}(${params}):\n` +
      this.tabString.repeat(this.handleIndentationLevel(true) + 1);

    return f;
  }

  generateFunctionCall(
    name: string,
    args: { name?: string; value: any }[]
  ): string {
    const params = args
      .map((p) => (p.name ? `${p.name} = ${p.value}` : `${p.value}`))
      .join(", ");
    return (
      `${name}(${params}) \n` +
      this.tabString.repeat(this.handleIndentationLevel())
    );
  }

  generateReturn(value?: string): string {
    return (
      `return ${value ?? ""} ` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
  }

  declareClass(
    name: string,
    properties?: { name: string; type?: string }[],
    methods?: {
      name: string;
      parameters: { name: string; value?: any }[];
      body?: string[];
    }[]
  ): string {
    if (!this.isValidVariableName(name)) {
      throw new Error(`Invalid class name: ${name} `);
    }

    // check if valid properties names
    if (properties?.some((p) => !this.isValidVariableName(p.name))) {
      throw new Error(`Invalid property name`);
    }

    // check if valid method names
    if (methods?.some((m) => !this.isValidVariableName(m.name))) {
      throw new Error(`Invalid method name`);
    }

    let code = "";
    // add class name, capitalize first letter and swap spaces with underScores

    let className: string =
      name.charAt(0).toUpperCase() + name.slice(1).replace(/ /g, "_");
    let firstIndentationLevel = this.handleIndentationLevel();

    code += `${this.tabString.repeat(
      firstIndentationLevel
    )}class ${className}: \n`;

    // add constructor
    code += `${this.tabString.repeat(firstIndentationLevel + 1)}${
      this.tabString
    }def __init__(self, ${
      properties ? properties.map((p) => p.name).join(", ") : ""
    }): \n`;

    // add properties
    properties?.forEach((p) => {
      code += `${this.tabString}${this.tabString}self.${p.name} = ${p.name}\n`;
    });

    // add methods
    methods?.forEach((m) => {
      // sort the parameters so that the one's without value come first
      m.parameters.sort((a, b) => (a.value === undefined ? -1 : 1));
      const params = m.parameters
        .map(
          (p) =>
            `${p.name}${
              p.value
                ? ` = ${typeof p.value === "string" ? `"${p.value}"` : p.value}`
                : ""
            }`
        )
        .join(", ");
      // code += "\n";
      code += `\n${this.tabString}def ${m.name}(self, ${params}):\n${this.tabString}`;
      code += this.wrapInCodeBlock(m.body ?? ["pass\n"]);
    });

    return code + "\n" + this.tabString.repeat(firstIndentationLevel + 3);
  }

  /**
   * Import modules
   **/
  generateImportModule(library: string, modules: string[]): string {
    return (
      `from ${library} import ${modules.join(", ")}\n` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
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
  generateIfElse(
    condition: string,
    ifBody: string[],
    elseBody?: string[]
  ): string {
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
  generateForLoop(type: ForLoop, params: any, body?: string[]): string {
    switch (type) {
      case ForLoop.Range:
        return this.generateRangeLoop(params);
      case ForLoop.Iterable:
        return this.generateIterableLoop(params);
      case ForLoop.Enumerate:
        return this.generateEnumerateLoop(params);
      default:
        return `for `;
    }
  }

  generateIterableLoop(
    params: { iterators: string[]; iterable: string },
    body?: string[]
  ): string {
    const loopCode = `for ${params.iterators.join(", ")} in ${
      params.iterable
    }: \n${this.wrapInCodeBlock(body ?? [""])} `;
    return loopCode;
  }

  generateRangeLoop(
    params: { iterators: string[]; start: number; end: number; step?: number }
    // body?: string[]
  ): string {
    const { iterators, start, end, step } = params;

    let forLoop = "";
    let actualStart = start ?? 0;
    let actualEnd = end ?? 0;
    let actualStep = step ?? 1;

    let currentIndentationLevel = this.handleIndentationLevel(false);
    forLoop += this.tabString.repeat(currentIndentationLevel);

    if (actualStart < actualEnd) {
      forLoop = `for ${iterators.join(
        ", "
      )} in range(${actualStart}, ${actualEnd}, ${actualStep}): \n`;
    } else {
      forLoop = `for ${iterators.join(
        ", "
      )} in range(${actualStart}, ${actualEnd}, ${-actualStep}): \n`;
    }

    return forLoop + this.tabString.repeat(currentIndentationLevel + 1);
  }

  generateEnumerateLoop(
    params: { iterators: string[]; iterable: string; start?: number },
    body?: string[]
  ): string {
    const { iterators, iterable, start } = params;
    let currentIndentationLevel = this.handleIndentationLevel();
    // this.tabString.repeat(currentIndentationLevel);

    return (
      `${this.tabString.repeat(currentIndentationLevel)}for ${iterators.join(
        ", "
      )} in enumerate(${iterable}${start ? ` ,${start}` : ""}): \n` +
      this.tabString.repeat(currentIndentationLevel + 1)
    );
  }

  generateWhileLoop(condition: Condition[], body?: string[]): string {
    const conditionCode = condition
      .map(
        (c) => `${c.logicalOperator ?? ""} ${c.left} ${c.operator} ${c.right}`
      )
      .join(" ");

    let currentIndentationLevel = this.handleIndentationLevel();

    const loopCode = `${this.tabString.repeat(
      currentIndentationLevel
    )}while ${conditionCode}: \n${this.tabString.repeat(
      currentIndentationLevel + 1
    )}`;
    return loopCode;
  }

  /**
   * Try Except
   */
  generateTryExcept(
    tryBody: string[],
    exception: string,
    exceptionInstance: string,
    exceptBody: string[]
  ): string {
    const tryCode = `try: \n${this.wrapInCodeBlock(tryBody ?? [""])} `;
    const exceptCode = `except ${exception} as ${exceptionInstance}: \n${this.wrapInCodeBlock(
      exceptBody ?? [""]
    )} `;
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
  generateAssertion(variable: string, value: any, type: string): string {
    if (
      !Object.values(AssertionOperators).includes(type as AssertionOperators)
    ) {
      throw new Error(`Invalid assertion type: ${type}`);
    }
    switch (type) {
      case AssertionOperators.Equal:
        return (
          `assert ${variable} == ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssertionOperators.NotEqual:
        return (
          `assert ${variable} != ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssertionOperators.GreaterThanOrEqual:
        return (
          `assert ${variable} >= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case AssertionOperators.LessThanOrEqual:
        return (
          `assert ${variable} <= ${value}\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      default:
        throw new Error(`Invalid assertion type: ${type}`);
    }
  }

  /**
   * Generate Casting
   **/
  generateCasting(variable: any, type: string): string {
    // this.typemappings
    if (
      !Object.keys(this.typeMappings).includes(
        type.toLowerCase() as CastingTypes
      )
    ) {
      throw new Error(`Invalid casting type: ${type}`);
    }
    return (
      `${variable} = ${this.typeMappings[type]}(${variable})\n` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
  }

  /**
   * Generate User Input
   **/
  generateUserInput(variable: string, message?: string | undefined): string {
    return (
      `${variable} = input('${message ? message : ""}')\n` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
  }

  /**
   * Generate Print
   **/
  generatePrint(value: any, type?: string): string {
    switch (type) {
      case "string":
        return (
          `print("${value}")\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      case "variable":
        return (
          `print(${value})\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
      default:
        return (
          `print(${value})\n` +
          this.tabString.repeat(this.handleIndentationLevel(true))
        );
    }
  }

  /**
   * Read file
   **/
  //TODO: add options to read line, read all file, read character
  generateReadFile(path: string, variable: any): string {
    return (
      `${variable} = open("${path}", 'r').read()\n` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
  }

  /**
   * Write file
   **/
  generateWriteFile(path: string, content: any): string {
    return (
      `open("${path}", 'w').write("${content}")\n` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
  }

  /**
   * White spaces
   */
  addWhiteSpace(type: Whitespace, count?: number): string {
    let ws;
    switch (type) {
      case Whitespace.Space:
        ws = " ";
        break;
      case Whitespace.Tab:
        ws = "    ";
        break;
      case Whitespace.NewLine:
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
  generateLineComment(content: string): string {
    return (
      `# ${content}\n` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
  }

  generateBlockComment(content: string[]): string {
    return (
      `''' ${content.join("\n")} '''\n` +
      this.tabString.repeat(this.handleIndentationLevel(true))
    );
  }

  generateOperation(left: string, operator: Operator, right: string): string {
    return (
      `${left} ${operator} ${right} \n` + this.handleIndentationLevel(true)
    );
  }

  generateConditional(
    conditions: {
      keyword: "if" | "else" | "elif";
      condition?: Condition[];
      body?: string[];
    }[]
  ): string {
    let code = "";
    conditions.forEach((c) => {
      if (c.keyword === "if" || c.keyword === "elif") {
        code += `${c.keyword} ${c.condition
          ?.map(
            (cond) =>
              `${cond.logicalOperator ?? ""} ${cond.left} ${
                cond?.operator ?? "=="
              } ${cond.right}`
          )
          .join(" ")}: \n`;
      } else {
        code += `\nelse: \n`;
        // body
        // if (c.body) {
        //     code += `${this.wrapInCodeBlock(c.body)} `;
        // }
      }
    });
    return code;
  }

  generateArrayOperation(name: string, operation: string): string {
    // pack, uno
    return "";
  }

  exitScope(): string {
    let indentationLevel = this.handleIndentationLevel(false);
    return `\n${this.addWhiteSpace(
      Whitespace.Tab,
      Math.max(indentationLevel - 1, 0)
    )}`.repeat(2);
  }
}
