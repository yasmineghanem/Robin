"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
const vscode = __importStar(require("vscode"));
const code_1 = require("../constants/code");
const pythonCodeGenerator_1 = require("../code generation/pythonCodeGenerator");
const utilities_1 = require("../communication/utilities");
const constants_1 = require("../constants/constants");
// utilities
const getCurrentPosition = (editor) => {
    const position = editor.selection.active;
    return position;
};
const getFileExtension = (editor) => {
    return editor.document.fileName.split(".").pop();
};
const handleFailure = (message) => {
    (0, utilities_1.showError)(message);
    return {
        success: false,
        message: message,
    };
};
const handleSuccess = (message) => {
    (0, utilities_1.showMessage)(message);
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
    vscode.commands.registerCommand(code_1.DECLARE_VARIABLE, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            let s = await editor.edit((editBuilder) => {
                editBuilder.insert(getCurrentPosition(editor), codeGenerator.declareVariable(args.name, args.type, args.value));
            });
            console.log("ay haga");
            if (!s) {
                return handleFailure(code_1.VARIABLE_FAILURE);
            }
            return handleSuccess(code_1.VARIABLE_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
const assignVariable = () => {
    vscode.commands.registerCommand(code_1.ASSIGN_VARIABLE, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            let codeGenerator;
            switch (getFileExtension(editor)) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.assignVariable(args.name, args.value, args.type));
                });
                if (!s) {
                    return handleFailure(code_1.ASSIGNMENT_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.ASSIGNMENT_FAILURE);
            }
            return handleSuccess(code_1.ASSIGNMENT_SUCCESS);
        }
    });
};
// declare constant
const declareConstant = () => {
    vscode.commands.registerCommand(code_1.DECLARE_CONSTANT, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.declareConstant(args.name, args.value));
                });
                if (!s) {
                    return handleFailure(code_1.VARIABLE_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.VARIABLE_FAILURE);
            }
            return handleSuccess(code_1.VARIABLE_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
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
    vscode.commands.registerCommand(code_1.DECLARE_FUNCTION, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            try {
                const code = codeGenerator.declareFunction(args.name, args.parameters ?? [], args?.body);
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), code);
                });
                if (!s) {
                    return handleFailure(code_1.FUNCTION_FAILURE);
                }
                return handleSuccess(code_1.FUNCTION_SUCCESS);
            }
            catch (e) {
                return handleFailure(code_1.FUNCTION_FAILURE);
            }
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
const getAST = () => {
    vscode.commands.registerCommand(code_1.GET_AST, async (args) => {
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
            const data = await response.json();
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
    vscode.commands.registerCommand(code_1.FUNCTION_CALL, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            let s = await editor.edit((editBuilder) => {
                editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateFunctionCall(args.name, args.args));
            });
            if (!s) {
                return handleFailure(code_1.FUNCTION_CALL_FAILURE);
            }
            return handleSuccess(code_1.FUNCTION_CALL_SUCCESS);
        }
    });
};
// white space
const addWhiteSpace = () => {
    vscode.commands.registerCommand(code_1.ADD_WHITESPACE, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            let codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
            let s = await editor.edit((editBuilder) => {
                editBuilder.insert(getCurrentPosition(editor), codeGenerator.addWhiteSpace(args.type, args?.count));
            });
            if (!s) {
                return handleFailure(code_1.WHITE_SPACE_FAILURE);
            }
            return handleSuccess(code_1.WHITE_SPACE_SUCCESS);
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
    vscode.commands.registerCommand(code_1.FOR_LOOP, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateForLoop(args.type, args));
                });
                if (!s) {
                    return handleFailure(code_1.LOOP_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.LOOP_FAILURE);
            }
            return handleSuccess(code_1.LOOP_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
const whileLoop = () => {
    vscode.commands.registerCommand(code_1.WHILE_LOOP, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateWhileLoop(args.condition, args?.body));
                });
                if (!s) {
                    return handleFailure(code_1.LOOP_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.LOOP_FAILURE);
            }
            return handleSuccess(code_1.LOOP_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// operation
const operation = () => {
    vscode.commands.registerCommand(code_1.OPERATION, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateOperation(args.left, args.operator, args.right));
                });
                if (!s) {
                    return handleFailure(code_1.OPERATION_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.OPERATION_FAILURE);
            }
            return handleSuccess(code_1.OPERATION_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
//try except
const tryExcept = () => {
    vscode.commands.registerCommand(code_1.TRY_EXCEPT, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateTryExcept(args.tryBody, args.exception, args.exceptionInstance, args.exceptBody));
                });
                if (!s) {
                    return handleFailure(code_1.TRY_EXCEPT_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.TRY_EXCEPT_FAILURE);
            }
            return handleSuccess(code_1.TRY_EXCEPT_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// conditional
const conditional = () => {
    vscode.commands.registerCommand(code_1.CONDITIONAL, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateConditional(args));
                });
                if (!s) {
                    return handleFailure(code_1.OPERATION_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.OPERATION_FAILURE);
            }
            return handleSuccess(code_1.OPERATION_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
//Import
const importLibrary = () => {
    vscode.commands.registerCommand(code_1.IMPORT_LIBRARY, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateImportLibrary(args.library));
                });
                if (!s) {
                    return handleFailure(code_1.IMPORT_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.IMPORT_FAILURE);
            }
            return handleSuccess(code_1.IMPORT_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
//Import Module
const importModule = () => {
    vscode.commands.registerCommand(code_1.IMPORT_MODULE, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateImportModule(args.library, args?.modules));
                });
                if (!s) {
                    return handleFailure(code_1.IMPORT_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.IMPORT_FAILURE);
            }
            return handleSuccess(code_1.IMPORT_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// assertion
const assertion = () => {
    vscode.commands.registerCommand(code_1.ASSERTION, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateAssertion(args.variable, args.value, args.type));
                });
                if (!s) {
                    return handleFailure(code_1.ASSERTION_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.ASSERTION_FAILURE);
            }
            return handleSuccess(code_1.ASSERTION_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// casting
const typeCasting = () => {
    vscode.commands.registerCommand(code_1.TYPE_CASTING, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateCasting(args.variable, args.type));
                });
                if (!s) {
                    return handleFailure(code_1.CASTING_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.CASTING_FAILURE);
            }
            return handleSuccess(code_1.CASTING_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// array operations (pack, unpack, zip, unzip)
const arrayOperations = () => {
    vscode.commands.registerCommand(code_1.ARRAY_OPERATION, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateOperation(args.left, args.operator, args.right));
                });
                if (!s) {
                    return handleFailure(code_1.OPERATION_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.OPERATION_FAILURE);
            }
            return handleSuccess(code_1.OPERATION_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// user input
const userInput = () => {
    vscode.commands.registerCommand(code_1.USER_INPUT, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateUserInput(args.variable, args.message));
                });
                if (!s) {
                    return handleFailure(code_1.USER_INPUT_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.USER_INPUT_FAILURE);
            }
            return handleSuccess(code_1.USER_INPUT_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// print
const printConsole = () => {
    vscode.commands.registerCommand(code_1.PRINT, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generatePrint(args.variable, args.type));
                });
                if (!s) {
                    return handleFailure(code_1.PRINT_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.PRINT_FAILURE);
            }
            return handleSuccess(code_1.PRINT_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// line comment
const lineComment = () => {
    vscode.commands.registerCommand(code_1.LINE_COMMENT, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateLineComment(args.content));
                });
                if (!s) {
                    return handleFailure(code_1.LINE_COMMENT_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.LINE_COMMENT_FAILURE);
            }
            return handleSuccess(code_1.LINE_COMMENT_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// block comment
const blockComment = () => {
    vscode.commands.registerCommand(code_1.BLOCK_COMMENT, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateBlockComment(args.content));
                });
                if (!s) {
                    return handleFailure(code_1.BLOCK_COMMENT_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.BLOCK_COMMENT_FAILURE);
            }
            return handleSuccess(code_1.BLOCK_COMMENT_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// read from file
const readFile = () => {
    vscode.commands.registerCommand(code_1.READ_FILE, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateReadFile(args.path, args.variable));
                });
                if (!s) {
                    return handleFailure(code_1.READ_FILE_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.READ_FILE_FAILURE);
            }
            return handleSuccess(code_1.READ_FILE_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
// write to file
const writeFile = () => {
    vscode.commands.registerCommand(code_1.WRITE_FILE, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            // try catch
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.generateWriteFile(args.path, args.content));
                });
                if (!s) {
                    return handleFailure(code_1.WRITE_FILE_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.WRITE_FILE_FAILURE);
            }
            return handleSuccess(code_1.WRITE_FILE_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
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
    vscode.commands.registerCommand(code_1.DECLARE_CLASS, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.declareClass(args?.name, args?.properties, args?.methods));
                });
                if (!s) {
                    return handleFailure(code_1.FUNCTION_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(code_1.FUNCTION_FAILURE);
            }
            return handleSuccess(code_1.FUNCTION_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
    });
};
const exitScope = () => {
    vscode.commands.registerCommand(code_1.EXIT_SCOPE, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);
            let codeGenerator;
            switch (ext) {
                case constants_1.EXTENSIONS.PYTHON:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                case constants_1.EXTENSIONS.JUPYTER:
                    codeGenerator = new pythonCodeGenerator_1.PythonCodeGenerator(editor);
                    break;
                default:
                    return handleFailure(code_1.FILE_EXT_FAILURE);
            }
            try {
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), codeGenerator.exitScope());
                });
                if (!s) {
                    return handleFailure(code_1.EXIT_SCOPE_SUCCESS);
                }
            }
            catch (e) {
                return handleFailure(code_1.EXIT_SCOPE_FAILURE);
            }
            return handleSuccess(code_1.EXIT_SCOPE_SUCCESS);
        }
        return handleFailure(code_1.NO_ACTIVE_TEXT_EDITOR);
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
exports.default = registerCodeCommands;
//# sourceMappingURL=codeCommands.js.map