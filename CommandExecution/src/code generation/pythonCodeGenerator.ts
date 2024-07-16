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
  private editor: vscode.TextEditor;
  private typeMappings: { [key: string]: string };
  private operatorMappings: { [key: string]: string };
  private reservedKeywords: Array<string>;

  // constructor
  constructor(editor: vscode.TextEditor) {
    super();
    this.reservedKeywords = pythonSpecifics.reservedKeywords;
    this.typeMappings = pythonSpecifics.typeMappings;
    this.operatorMappings = pythonSpecifics.operatorMappings;
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

  /**
   * wrap code in a code block with '`' character
   **/
  wrapInCodeBlock(lines: string[]): string {
    return lines.map((line) => `    ${line}`).join("\n");
  }

  //********************************************//
  /**
   * Declare variables
   **/
  declareVariable(name: string, type?: string, initialValue?: any): string {
    if (!this.isValidVariableName(name)) {
      throw new Error(`Invalid variable name: ${name}`);
    }
    let code = "";
    // replace spaces with underscores
    let v_name = name.replace(/\s+/g,"_");
    if (type) {
      let mappedType: string =
        this.typeMappings[type as keyof typeof this.typeMappings];

      if (initialValue) {
        code = `${v_name }: ${mappedType} = ${initialValue}`;
      }

      code = `${v_name }: ${mappedType}`;
    } else if (initialValue !== undefined) {
      code = `${v_name } = ${initialValue}`;
    } else {
      code = `${v_name}`;
    }

    this.handleScope(code);
    return code;
  }

  /**
   * Declare constants
   **/
  declareConstant(name: string, value: any): string {
    if (!this.isValidConstantName(name.toUpperCase())) {
      throw new Error(`Invalid constant name: ${name}`);
    }
    const code = `${name.toUpperCase()} = ${value}`;
    this.handleScope(code);
    return code;
  }

  /**
   * Assign variables
   **/
  assignVariable(name: string, value: any, type: string): string {
    //Check before if RHS is same type as LHS
    ///////// we need function to check the type of the variable /////////
    if (!Object.keys(this.operatorMappings).includes(type)) {
      throw new Error(`Invalid assignment type: ${type}`);
    }

    let code = `${name} ${this.operatorMappings[type]} ${value}`;
    this.handleScope(code);
    return code;
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
    const f = `def ${name}(${params}):`;

    this.handleScope(f);

    return f;
  }

  generateFunctionCall(
    name: string,
    args: { name?: string; value: any }[]
  ): string {
    const params = args
      .map((p) => (p.name ? `${p.name} = ${p.value}` : `${p.value}`))
      .join(", ");
    const code = `${name}(${params})`;

    this.handleScope(code);
    return code;
  }

  generateReturn(value?: string): string {
    const code = `return ${value ?? ""}`;
    this.handleScope(code);
    return code;
  }

  async declareClass(
    name: string,
    properties?: { name: string; type?: string }[],
    methods?: {
      name: string;
      parameters: { name: string; value?: any }[];
      body?: string[];
    }[]
  ): Promise<string> {
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
    // let firstIndentationLevel = this.handleIndentationLevel();

    code = `class ${className}:`;
    await this.handleScope(code);

    // add constructor
    code = `def __init__(self, ${
      properties ? properties.map((p) => p.name).join(", ") : ""
    }):`;
    await this.handleScope(code);

    // add properties
    properties?.forEach(async (p) => {
      code = `self.${p.name} = ${p.name}`;
      await this.handleScope(code);
    });

    // add methods
    methods?.forEach(async (m) => {
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

      code = `def ${m.name}(self, ${params}):`;
      await this.handleScope(code);
    });

    return code;
  }

  /**
   * Import modules
   **/
  generateImportModule(library: string, modules: string[]): string {
    let code = `from ${library} import ${modules.join(", ")}`;
    this.handleScope(code);

    return code;
  }

  generateImportLibrary(library: string): string {
    let code = `import ${library}`;
    this.handleScope(code);

    return code;
  }

  /**
   * Conditional statements
   * if, if-else
   **/
  generateIf(condition: string, body: string[]): string {
    const ifCode = `if ${condition}:`;
    this.handleScope(ifCode);

    return ifCode;
  }
  generateIfElse(
    condition: string,
    ifBody: string[],
    elseBody?: string[]
  ): string {
    return ``;
  }

  /**
   * Loop statements
   * for, while, do-while
   **/

  generateForLoop(type: ForLoop, params: any, body?: string[]): string {
    let code = "";
    switch (type) {
      case ForLoop.Range:
        code = this.generateRangeLoop(params);
        break;
      case ForLoop.Iterable:
        code = this.generateIterableLoop(params);
        break;
      case ForLoop.Enumerate:
        code = this.generateEnumerateLoop(params);
        break;
      default:
        code = `for :`;
        break;
    }

    this.handleScope(code);
    return code;
  }

  generateIterableLoop(
    params: { iterators: string[]; iterable: string },
    body?: string[]
  ): string {
    const loopCode = `for ${params.iterators.join(", ")} in ${
      params.iterable
    }:`;

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

    if (actualStart < actualEnd) {
      forLoop = `for ${iterators.join(
        ", "
      )} in range(${actualStart}, ${actualEnd}, ${actualStep}):`;
    } else {
      forLoop = `for ${iterators.join(
        ", "
      )} in range(${actualStart}, ${actualEnd}, ${-actualStep}):`;
    }

    return forLoop;
  }

  generateEnumerateLoop(
    params: { iterators: string[]; iterable: string; start?: number },
    body?: string[]
  ): string {
    const { iterators, iterable, start } = params;
    // let currentIndentationLevel = this.handleIndentationLevel();
    // this.tabString.repeat(currentIndentationLevel);

    let code = `for ${iterators.join(", ")} in enumerate(${iterable}${
      start ? ` ,${start}` : ""
    }):`;

    return code;
  }

  generateWhileLoop(condition: Condition[], body?: string[]): string {
    const conditionCode = condition
      .map(
        (c) =>
          `${
            c.logicalOperator
              ? `${this.operatorMappings[c.logicalOperator] ?? "=="} `
              : ""
          } ${c.left} ${this.operatorMappings[c.operator] ?? "=="} ${c.right}`
      )
      .join(" ");

    let loopCode = `while ${conditionCode}:`;
    this.handleScope(loopCode);
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

  generateAssertion(variable: string, value: any, type: string): string {
    if (!Object.keys(this.operatorMappings).includes(type)) {
      throw new Error(`Invalid assertion type: ${type}`);
    }
    let code = `assert ${variable} ${this.operatorMappings[type]} ${value}`;
    this.handleScope(code);
    return code;
  }

  /**
   * Generate Casting
   **/
  generateCasting(variable: any, type: string): string {
    // this.typemappings
    if (!Object.keys(this.operatorMappings).includes(type as CastingTypes)) {
      throw new Error(`Invalid casting type: ${type}`);
    }
    let code = `${variable} = ${this.operatorMappings[type]}(${variable})`;
    this.handleScope(code);
    return code;
  }

  /**
   * Generate User Input
   **/
  generateUserInput(variable: string, message?: string | undefined): string {
    let code = `${variable} = input('${message ? message : ""}')`;
    this.handleScope(code);
    return code;
  }

  /**
   * Generate Print
   **/
  generatePrint(value: any, type?: string): string {
    let code = "";
    switch (type) {
      case "string":
        code = `print("${value}")`;

      case "variable":
        code = `print(${value})`;

      default:
        code = `print(${value})`;
    }

    this.handleScope(code);
    return code;
  }

  /**
   * Read file
   **/
  //TODO: add options to read line, read all file, read character
  generateReadFile(path: string, variable: any): string {
    let code = `${variable} = open("${path}", 'r').read()`;
    this.handleScope(code);
    return code;
  }

  /**
   * Write file
   **/
  generateWriteFile(path: string, content: any): string {
    let code = `open("${path}", 'w').write("${content}")`;
    this.handleScope(code);
    return code;
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
    this.insertCode(ws.repeat(count ?? 1));

    return "alo";
  }

  /**
   * Comments
   * Single line comments
   * Multi line comments
   **/
  generateLineComment(content: string): string {
    let code = `# ${content}`;
    this.handleScope(code);
    return code;
  }

  generateBlockComment(content: string[]): string {
    return `''' ${content.join("\n")} '''\n`;
  }

  generateOperation(left: string, operator: string, right: string): string {
    let code = `${left} ${this.operatorMappings[operator] ?? "=="} ${right}`;
    this.handleScope(code);
    return code;
  }

  async generateConditional(
    conditions: {
      keyword: "if" | "else" | "elif";
      condition?: Condition[];
      body?: string[];
    }[]
  ): Promise<string> {
    let code = "";

    for (let c of conditions) {
      if (c.keyword === "if" || c.keyword === "elif") {
        code = `${c.keyword} ${c.condition
          ?.map(
            (cond) =>
              `${
                cond.logicalOperator
                  ? `${this.operatorMappings[cond.logicalOperator] ?? "=="} `
                  : ""
              } ${cond.left} ${this.operatorMappings[cond.operator] ?? "=="} ${
                cond.right
              }`
          )
          .join(" ")}:`;
      } else {
        code = `else:`;
      }
      if (c.keyword === "elif" || c.keyword === "else") {
        await this.exitScope();
      }
      await this.handleScope(code);
    }

    return code;
  }

  exitScope() {
    // outdent
    return new Promise<void>((resolve, reject) => {
      vscode.commands.executeCommand("outdent").then(
        () => {
          resolve();
        },
        (err) => {
          reject(err);
        }
      );
    });
  }

  handleScope(code: string) {
    return new Promise<void>((resolve, reject) => {
      this.insertCode(code).then(() => {
        vscode.commands.executeCommand("editor.action.insertLineAfter").then(
          () => {
            resolve();
          },
          (err) => {
            reject(err);
          }
        );
      });
    });
  }

  insertCode(code: string) {
    return this.editor.edit(
      (editBuilder) => {
        editBuilder.insert(this.editor.selection.active, code);
      }
      // { undoStopBefore: true, undoStopAfter: false }
    );
  }
}
