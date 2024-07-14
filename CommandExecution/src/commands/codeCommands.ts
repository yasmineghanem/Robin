import * as vscode from "vscode";
import { PythonCodeGenerator } from "../code generation/pythonCodeGenerator";
import { showError, showMessage } from "../communication/utilities";
import {
  ADD_WHITESPACE,
  ARRAY_OPERATION,
  ASSERTION,
  ASSERTION_FAILURE,
  ASSERTION_SUCCESS,
  ASSIGN_VARIABLE,
  ASSIGNMENT_FAILURE,
  ASSIGNMENT_SUCCESS,
  BLOCK_COMMENT,
  BLOCK_COMMENT_FAILURE,
  BLOCK_COMMENT_SUCCESS,
  CASTING_FAILURE,
  CASTING_SUCCESS,
  CLASS_FAILURE,
  CLASS_SUCCESS,
  CONDITIONAL,
  DECLARE_CLASS,
  DECLARE_CONSTANT,
  DECLARE_FUNCTION,
  DECLARE_VARIABLE,
  EXIT_SCOPE,
  EXIT_SCOPE_FAILURE,
  EXIT_SCOPE_SUCCESS,
  FILE_EXT_FAILURE,
  FOR_LOOP,
  FUNCTION_CALL,
  FUNCTION_CALL_FAILURE,
  FUNCTION_CALL_SUCCESS,
  FUNCTION_FAILURE,
  FUNCTION_SUCCESS,
  GET_AST,
  IMPORT_FAILURE,
  IMPORT_LIBRARY,
  IMPORT_MODULE,
  IMPORT_SUCCESS,
  LINE_COMMENT,
  LINE_COMMENT_FAILURE,
  LINE_COMMENT_SUCCESS,
  LOOP_FAILURE,
  LOOP_SUCCESS,
  NO_ACTIVE_TEXT_EDITOR,
  OPERATION,
  OPERATION_FAILURE,
  OPERATION_SUCCESS,
  PRINT,
  PRINT_FAILURE,
  PRINT_SUCCESS,
  READ_FILE,
  READ_FILE_FAILURE,
  READ_FILE_SUCCESS,
  TRY_EXCEPT,
  TRY_EXCEPT_FAILURE,
  TRY_EXCEPT_SUCCESS,
  TYPE_CASTING,
  USER_INPUT,
  USER_INPUT_FAILURE,
  USER_INPUT_SUCCESS,
  VARIABLE_FAILURE,
  VARIABLE_SUCCESS,
  WHILE_LOOP,
  WHITE_SPACE_FAILURE,
  WHITE_SPACE_SUCCESS,
  WRITE_FILE,
  WRITE_FILE_FAILURE,
  WRITE_FILE_SUCCESS,
} from "../constants/code";
import { EXTENSIONS } from "../constants/constants";

// utilities
const getFileExtension = (editor: vscode.TextEditor): string | undefined => {
  return editor.document.fileName.split(".").pop();
};

const handleFailure = (message: string): any => {
  showError(message);
  return {
    success: false,
    message: message,
  };
};

const handleSuccess = (message: string): any => {
  showMessage(message);
  return {
    success: true,
    message: message,
  };
};

// {
//     "name": "x",
//     "value" : 0
// }
// declare variable
const declareVariable = () => {
  
  vscode.commands.registerCommand(DECLARE_VARIABLE, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;
      
      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      let s = codeGenerator.declareVariable(args.name, args.type, args.value);
      
      if (!s) {
        return handleFailure(VARIABLE_FAILURE);
      }

      return handleSuccess(VARIABLE_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

const assignVariable = () => {
  vscode.commands.registerCommand(ASSIGN_VARIABLE, async (args) => {
    const editor = vscode.window.activeTextEditor;

    if (editor) {
      let codeGenerator;

      switch (getFileExtension(editor)) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      try {
        let s = codeGenerator.assignVariable(args.name, args.value, args.type);
        if (!s) {
          return handleFailure(ASSIGNMENT_FAILURE);
        }
      } catch (e) {
        return handleFailure(ASSIGNMENT_FAILURE);
      }

      return handleSuccess(ASSIGNMENT_SUCCESS);
    }
  });
};

// declare constant
const declareConstant = () => {
  vscode.commands.registerCommand(DECLARE_CONSTANT, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      try {
        let s = codeGenerator.declareConstant(args.name, args.value);

        if (!s) {
          return handleFailure(VARIABLE_FAILURE);
        }
      } catch (e) {
        return handleFailure(VARIABLE_FAILURE);
      }

      return handleSuccess(VARIABLE_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// declare function
// {
//     "name": "x_string",
//     "paramaters": [
//         {
//             "name": "x",
//             "value": "test"
//         },
//         {
//             "name": "y",
//             "value": "test"
//         }
//     ]
// }

const declareFunction = () => {
  vscode.commands.registerCommand(DECLARE_FUNCTION, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      try {
        let s = codeGenerator.declareFunction(
          args.name,
          args.parameters ?? [],
          args?.body
        );

        if (!s) {
          return handleFailure(FUNCTION_FAILURE);
        }

        return handleSuccess(FUNCTION_SUCCESS);
      } catch (e) {
        return handleFailure(FUNCTION_FAILURE);
      }
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

const getAST = () => {
  vscode.commands.registerCommand(GET_AST, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      const text = editor.document.getText();

      // get the corresponding AST from the microservice
      const response = await fetch("http://localhost:2806/ast", {
        method: "POST",
        body: JSON.stringify({ code: text }),
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data: any = await response.json();
      // console.log();
      console.log("ROBIN FILTER \n" + data);

      return {
        success: true,
        message: "AST retrieved successfully",
        data: data,
      };
    }
  });
};

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

// function call
const functionCall = () => {
  vscode.commands.registerCommand(FUNCTION_CALL, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      let s = codeGenerator.generateFunctionCall(args.name, args.args);

      if (!s) {
        return handleFailure(FUNCTION_CALL_FAILURE);
      }

      return handleSuccess(FUNCTION_CALL_SUCCESS);
    }
  });
};

// white space
const addWhiteSpace = () => {
  vscode.commands.registerCommand(ADD_WHITESPACE, async (args) => {
    const editor = vscode.window.activeTextEditor;

    if (editor) {
      let codeGenerator = new PythonCodeGenerator(editor);
      let s =
        // await editor.edit((editBuilder) => {
        //   editBuilder.insert(
        //     getCurrentPosition(editor),
        codeGenerator.addWhiteSpace(args.type, args?.count);
      //   );
      // });

      if (!s) {
        return handleFailure(WHITE_SPACE_FAILURE);
      }

      return handleSuccess(WHITE_SPACE_SUCCESS);
    }
  });
};

// {
//     "type": "range",
//     "iterator": [
//         "i"
//     ],
//     "start": "1",
//     "end": "5"
//     //   "step": ""
// }
const forLoop = () => {
  vscode.commands.registerCommand(FOR_LOOP, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateForLoop(args.type, args);

        if (!s) {
          return handleFailure(LOOP_FAILURE);
        }
      } catch (e) {
        return handleFailure(LOOP_FAILURE);
      }

      return handleSuccess(LOOP_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

const whileLoop = () => {
  vscode.commands.registerCommand(WHILE_LOOP, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        const s = codeGenerator.generateWhileLoop(args.condition, args?.body);

        if (!s) {
          return handleFailure(LOOP_FAILURE);
        }
      } catch (e) {
        return handleFailure(LOOP_FAILURE);
      }

      return handleSuccess(LOOP_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// operation
const operation = () => {
  vscode.commands.registerCommand(OPERATION, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateOperation(
          args.left,
          args.operator,
          args.right
        );

        if (!s) {
          return handleFailure(OPERATION_FAILURE);
        }
      } catch (e) {
        return handleFailure(OPERATION_FAILURE);
      }

      return handleSuccess(OPERATION_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

//try except
const tryExcept = () => {
  vscode.commands.registerCommand(TRY_EXCEPT, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateTryExcept(
          args.tryBody,
          args.exception,
          args.exceptionInstance,
          args.exceptBody
        );

        if (!s) {
          return handleFailure(TRY_EXCEPT_FAILURE);
        }
      } catch (e) {
        return handleFailure(TRY_EXCEPT_FAILURE);
      }

      return handleSuccess(TRY_EXCEPT_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// conditional
const conditional = () => {
  vscode.commands.registerCommand(CONDITIONAL, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateConditional(args);

        if (!s) {
          return handleFailure(OPERATION_FAILURE);
        }
      } catch (e) {
        return handleFailure(OPERATION_FAILURE);
      }

      return handleSuccess(OPERATION_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

//Import
const importLibrary = () => {
  vscode.commands.registerCommand(IMPORT_LIBRARY, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateImportLibrary(args.library);

        if (!s) {
          return handleFailure(IMPORT_FAILURE);
        }
      } catch (e) {
        return handleFailure(IMPORT_FAILURE);
      }

      return handleSuccess(IMPORT_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

//Import Module
const importModule = () => {
  vscode.commands.registerCommand(IMPORT_MODULE, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateImportModule(args.library, args?.modules);

        if (!s) {
          return handleFailure(IMPORT_FAILURE);
        }
      } catch (e) {
        return handleFailure(IMPORT_FAILURE);
      }

      return handleSuccess(IMPORT_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// assertion
const assertion = () => {
  vscode.commands.registerCommand(ASSERTION, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateAssertion(
          args.variable,
          args.value,
          args.type
        );

        if (!s) {
          return handleFailure(ASSERTION_FAILURE);
        }
      } catch (e) {
        return handleFailure(ASSERTION_FAILURE);
      }

      return handleSuccess(ASSERTION_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// casting
const typeCasting = () => {
  vscode.commands.registerCommand(TYPE_CASTING, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateCasting(args.variable, args.type);

        if (!s) {
          return handleFailure(CASTING_FAILURE);
        }
      } catch (e) {
        return handleFailure(CASTING_FAILURE);
      }

      return handleSuccess(CASTING_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// array operations (pack, unpack, zip, unzip)
const arrayOperations = () => {
  vscode.commands.registerCommand(ARRAY_OPERATION, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateOperation(
          args.left,
          args.operator,
          args.right
        );

        if (!s) {
          return handleFailure(OPERATION_FAILURE);
        }
      } catch (e) {
        return handleFailure(OPERATION_FAILURE);
      }

      return handleSuccess(OPERATION_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// user input
const userInput = () => {
  vscode.commands.registerCommand(USER_INPUT, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateUserInput(args.variable, args.message);

        if (!s) {
          return handleFailure(USER_INPUT_FAILURE);
        }
      } catch (e) {
        return handleFailure(USER_INPUT_FAILURE);
      }

      return handleSuccess(USER_INPUT_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// print
const printConsole = () => {
  vscode.commands.registerCommand(PRINT, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generatePrint(args.variable, args.type);

        if (!s) {
          return handleFailure(PRINT_FAILURE);
        }
      } catch (e) {
        return handleFailure(PRINT_FAILURE);
      }

      return handleSuccess(PRINT_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// line comment
const lineComment = () => {
  vscode.commands.registerCommand(LINE_COMMENT, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateLineComment(args.content);

        if (!s) {
          return handleFailure(LINE_COMMENT_FAILURE);
        }
      } catch (e) {
        return handleFailure(LINE_COMMENT_FAILURE);
      }

      return handleSuccess(LINE_COMMENT_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// block comment
const blockComment = () => {
  vscode.commands.registerCommand(BLOCK_COMMENT, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateBlockComment(args.content);

        if (!s) {
          return handleFailure(BLOCK_COMMENT_FAILURE);
        }
      } catch (e) {
        return handleFailure(BLOCK_COMMENT_FAILURE);
      }

      return handleSuccess(BLOCK_COMMENT_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// read from file
const readFile = () => {
  vscode.commands.registerCommand(READ_FILE, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateReadFile(args.path, args.variable);

        if (!s) {
          return handleFailure(READ_FILE_FAILURE);
        }
      } catch (e) {
        return handleFailure(READ_FILE_FAILURE);
      }

      return handleSuccess(READ_FILE_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};
// write to file
const writeFile = () => {
  vscode.commands.registerCommand(WRITE_FILE, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      // try catch
      try {
        let s = codeGenerator.generateWriteFile(args.path, args.content);

        if (!s) {
          return handleFailure(WRITE_FILE_FAILURE);
        }
      } catch (e) {
        return handleFailure(WRITE_FILE_FAILURE);
      }

      return handleSuccess(WRITE_FILE_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// {
//     "name": "testClass",
//     "methods": [
//         {
//             "name": "test_function",
//             "parameters": [
//                 {
//                     "name": "x_variable",
//                     "value": "test"
//                 },
//                 {
//                     "name": "y"
//                 }
//             ]
//         },
//         {
//             "name": "test_function",
//             "parameters": [
//                 {
//                     "name": "x_variable",
//                     "value": "test"
//                 },
//                 {
//                     "name": "y"
//                 }
//             ]
//         }
//     ]
// }

// declare class

const declareClass = () => {
  vscode.commands.registerCommand(DECLARE_CLASS, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      try {
        let s = codeGenerator.declareClass(
          args?.name,
          args?.properties,
          args?.methods
        );

        if (!s) {
          return handleFailure(CLASS_SUCCESS);
        }
      } catch (e) {
        return handleFailure(CLASS_FAILURE);
      }

      return handleSuccess(CLASS_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

const exitScope = () => {
  vscode.commands.registerCommand(EXIT_SCOPE, async (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
      // check for extension
      const ext = getFileExtension(editor);

      let codeGenerator;

      switch (ext) {
        case EXTENSIONS.PYTHON:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        case EXTENSIONS.JUPYTER:
          codeGenerator = new PythonCodeGenerator(editor);
          break;
        default:
          return handleFailure(FILE_EXT_FAILURE);
      }

      try {
        let s = codeGenerator.exitScope();

        if (!s) {
          return handleFailure(EXIT_SCOPE_SUCCESS);
        }
      } catch (e) {
        return handleFailure(EXIT_SCOPE_FAILURE);
      }

      return handleSuccess(EXIT_SCOPE_SUCCESS);
    }
    return handleFailure(NO_ACTIVE_TEXT_EDITOR);
  });
};

// register commands
const registerCodeCommands = () => {
  const commands = [
    declareVariable,
    declareFunction,
    getAST,
    functionCall,
    declareConstant,
    addWhiteSpace,
    forLoop,
    whileLoop,
    operation,
    conditional,
    assignVariable,
    importLibrary,
    importModule,
    arrayOperations,
    assertion,
    typeCasting,
    userInput,
    printConsole,
    lineComment,
    blockComment,
    readFile,
    writeFile,
    declareClass,
    tryExcept,
    exitScope,
  ];

  commands.forEach((command) => {
    command();
  });
};

export default registerCodeCommands;
